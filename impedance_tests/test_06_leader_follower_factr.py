#!/usr/bin/env python3
"""Test 06 — Leader-follower Cartesian impedance WITH FACTR friction compensation.

The leader's EE pose drives x_des; the follower tracks it with Cartesian impedance
(per-joint torque budget) + FACTR friction comp so the heavy shoulder/elbow break
loose and follow. See factr_friction.py for the friction law.

Control (follower, per cycle):
    tau_imp  = clip(J^T F, -tau_clip_vec, +tau_clip_vec)       # per-joint impedance clamp
    enable   = ramp 0->1 over --fric-ramp seconds after engage
    tau_fric = friction_torque(qdot, tau_imp, params, enable)   # breakaway + viscous comp
    tau_arm  = clip(tau_imp + tau_fric, -(tau_clip+fric_max), +(tau_clip+fric_max))
    command_joint_torque([tau_arm, gripper=0], kp=0 arm, kd=joint_damping, pos=hold gripper)

LOGGING: every run records a summary to --log (default factr_last_run.txt) with leader &
follower joint travel, EE tracking-error stats, and per-joint diagnostics (stuck% /
clamp% / oscillation) plus tuning suggestions -- so the run can be analysed from the file
without watching the arm.

OFFLINE (no hardware):  python test_06_leader_follower_factr.py --selftest
HARDWARE:
  python test_06_leader_follower_factr.py --follower-can can_follower --leader-can can_leader \
      --mu-c <from test_05> --mu-scale 0.5 --max-vel 3.0
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from factr_friction import (  # noqa: E402
    default_yam_params, default_tau_clip, friction_torque, check_budget, friction_curve_demo, GRAVITY_MAX,
)
from i2rt.robots.utils import GripperType  # noqa: E402

ARM_DOF = 6
JOINT_NAMES = ["base", "shoulder", "elbow", "wpitch", "wyaw", "wroll"]


class MjKin:
    """FK + 6x6 site Jacobian for the YAM arm (one model serves leader & follower)."""

    def __init__(self, xml_path: str, site: str = "grasp_site"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError(f"site '{site}' not found in {xml_path}")
        self.nv = self.model.nv

    def _set(self, q):
        self.data.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def fk(self, q):
        self._set(q)
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def jacobian(self, q):
        self._set(q)
        jacp = np.zeros((3, self.nv)); jacr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return np.vstack([jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]])


def parse_vec(s, n, name):
    parts = [float(x) for x in s.split(",") if x.strip() != ""]
    if len(parts) == 1:
        return np.full(n, parts[0])
    if len(parts) == n:
        return np.array(parts)
    raise ValueError(f"--{name} expects 1 or {n} values, got {len(parts)}")


def selftest(xml_path: str) -> int:
    print(f"[selftest] loading {xml_path}")
    kin = MjKin(xml_path)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        q = rng.uniform(-0.5, 0.5, ARM_DOF)
        J = kin.jacobian(q); p0, R0 = kin.fk(q)
        eps = 1e-6; Jfd = np.zeros((6, ARM_DOF))
        for i in range(ARM_DOF):
            dq = np.zeros(ARM_DOF); dq[i] = eps
            p1, R1 = kin.fk(q + dq)
            Jfd[:3, i] = (p1 - p0) / eps
            Jfd[3:, i] = Rotation.from_matrix(R1 @ R0.T).as_rotvec() / eps
        worst = max(worst, np.abs(J - Jfd).max())
    print(f"[selftest] Jacobian worst |J - J_fd| = {worst:.2e} -> {'PASS' if worst < 1e-3 else 'FAIL'}")
    print()
    friction_curve_demo()
    return 0 if worst < 1e-3 else 1


def write_summary(path, cfg, rec, tau_clip_vec, fric_on, ended):
    """Compute a human+machine readable run summary and write it to `path`."""
    L = []
    def emit(s=""):
        L.append(s)
    n = len(rec["t"])
    emit("================== FACTR leader-follower run summary ==================")
    emit(f"ended: {ended}    logged samples: {n}")
    if n < 5:
        emit("(too few samples for stats)")
        txt = "\n".join(L) + "\n"
        with open(path, "w") as f:
            f.write(txt)
        print("\n" + txt)
        return
    t = np.array(rec["t"]); dur = max(t[-1] - t[0], 1e-6)
    ql = np.array(rec["ql"]); qf = np.array(rec["qf"])
    xl = np.array(rec["xl"]); xf = np.array(rec["xf"])
    perr = np.array(rec["perr"]); rerr = np.array(rec["rerr"])
    ti = np.array(rec["ti"]); tf = np.array(rec["tf"]); qd = np.array(rec["qd"])
    track_cm = np.linalg.norm(perr, axis=1) * 100.0
    r2d = 180.0 / np.pi

    emit(f"duration: {dur:.1f} s    sample rate ~{n / dur:.0f} Hz")
    emit("")
    emit("CONFIG:")
    for k, v in cfg.items():
        emit(f"  {k} = {v}")
    emit("")
    emit("LEADER EE travel (cm):   x %5.1f  y %5.1f  z %5.1f" % tuple((xl.max(0) - xl.min(0)) * 100))
    emit("FOLLOWER EE travel (cm): x %5.1f  y %5.1f  z %5.1f" % tuple((xf.max(0) - xf.min(0)) * 100))
    emit("")
    emit("EE TRACKING ERROR (cm):  mean %.1f  median %.1f  p95 %.1f  max %.1f" % (
        track_cm.mean(), np.percentile(track_cm, 50), np.percentile(track_cm, 95), track_cm.max()))
    emit("  per-axis |pos err| (cm): x mean %.1f/max %.1f | y mean %.1f/max %.1f | z mean %.1f/max %.1f" % (
        np.abs(perr[:, 0]).mean() * 100, np.abs(perr[:, 0]).max() * 100,
        np.abs(perr[:, 1]).mean() * 100, np.abs(perr[:, 1]).max() * 100,
        np.abs(perr[:, 2]).mean() * 100, np.abs(perr[:, 2]).max() * 100))
    rn = np.linalg.norm(rerr, axis=1)
    emit("  orientation err (rad):   mean %.2f  max %.2f" % (rn.mean(), rn.max()))
    emit("")
    emit("PER-JOINT (arm):  lead_range/fol_range (deg) | tau_imp mean/max | tau_fric mean | stuck%% | clamp%% | osc/s")
    stuck = np.zeros(ARM_DOF); clamp = np.zeros(ARM_DOF); osc = np.zeros(ARM_DOF)
    for j in range(ARM_DOF):
        lr = (ql[:, j].max() - ql[:, j].min()) * r2d
        fr = (qf[:, j].max() - qf[:, j].min()) * r2d
        ai = np.abs(ti[:, j]); af = np.abs(tf[:, j]); ad = np.abs(qd[:, j])
        stuck[j] = float(np.mean((ai > 0.5) & (ad < 0.05)))      # commanded but not moving
        clamp[j] = float(np.mean(ai >= 0.95 * tau_clip_vec[j]))  # impedance at clamp
        sv = np.sign(qd[:, j]); sv[ad < 0.05] = 0; nz = sv[sv != 0]
        rev = int(np.sum(np.diff(nz) != 0)) if len(nz) > 1 else 0
        osc[j] = rev / dur
        emit("  J%d %-8s %5.1f / %5.1f | %.2f / %.2f | %.2f | %3.0f%% | %3.0f%% | %.1f" % (
            j, JOINT_NAMES[j], lr, fr, ai.mean(), ai.max(), af.mean(), stuck[j] * 100, clamp[j] * 100, osc[j]))
    # Hold stability: does the follower keep moving while the leader is held still?
    leadspd = np.linalg.norm(np.diff(xl, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)
    qd_al = np.abs(qd[1:]); still = leadspd < 0.02
    emit("")
    if int(still.sum()) >= 5:
        qd_still = qd_al[still]
        moving_frac = float(np.mean(np.max(qd_still, axis=1) > 0.1))
        pj_hold = qd_still.mean(axis=0)
        emit("HOLD STABILITY (leader <2 cm/s for %.0f%% of run):" % (100 * still.mean()))
        emit("  follower still moving (any arm joint >0.1 rad/s): %.0f%% of held samples" % (100 * moving_frac))
        emit("  mean follower |joint vel| while held (rad/s): " +
             "  ".join("%s %.2f" % (JOINT_NAMES[j], pj_hold[j]) for j in range(ARM_DOF)))
    else:
        moving_frac = 0.0; pj_hold = np.zeros(ARM_DOF)
        emit("HOLD STABILITY: not enough leader-still time to assess (hold the leader still a few times next run)")
    emit("")
    emit("TUNING SUGGESTIONS:")
    sugg = []
    if int(still.sum()) >= 5 and moving_frac > 0.25:
        w = int(np.argmax(pj_hold))
        fix = ("raise its kd / lower K_rot" if w >= 3 else "lower mu_c or raise --v-eps")
        sugg.append(f"  HOLD-STABILITY: follower moves {moving_frac*100:.0f}% of the time the leader is held still "
                    f"(most active: J{w} {JOINT_NAMES[w]}) -> {fix}")
    for j in range(ARM_DOF):
        if stuck[j] > 0.15:
            extra = " (also at clamp -> raise tau_clip/K)" if clamp[j] > 0.2 else ""
            sugg.append(f"  J{j} {JOINT_NAMES[j]}: STUCK {stuck[j]*100:.0f}% (torque commanded, joint not moving) "
                        f"-> raise mu_c[{j}] (more friction comp){extra}")
        elif clamp[j] > 0.25:
            sugg.append(f"  J{j} {JOINT_NAMES[j]}: impedance SATURATING clamp {clamp[j]*100:.0f}% "
                        f"-> raise tau_clip[{j}] or K (more authority)")
        if fric_on and osc[j] > 3.0 and tf[:, j].std() > 0.05:
            sugg.append(f"  J{j} {JOINT_NAMES[j]}: OSCILLATING {osc[j]:.1f} rev/s with active friction comp "
                        f"-> lower mu_c[{j}] (likely over-compensated)")
    p95 = np.percentile(track_cm, 95)
    if not sugg:
        if p95 < 4:
            sugg.append("  Tracking looks GOOD: no joint stuck/saturating/oscillating, EE p95 < 4 cm. Gains OK.")
        else:
            sugg.append(f"  EE p95 {p95:.1f} cm is a bit high but no joint is clearly stuck -- "
                        f"raise K_trans for snappier tracking, or move the leader slower.")
    for s in sugg:
        emit(s)
    emit("=" * 70)
    txt = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(txt)
    print("\n" + txt)
    print(f"[summary written to {path}]")


def main() -> int:
    ap = argparse.ArgumentParser(description="YAM leader-follower Cartesian impedance + FACTR friction comp")
    ap.add_argument("--selftest", action="store_true", help="offline Jacobian + friction-curve check, no CAN")
    ap.add_argument("--follower-can", default="can_follower")
    ap.add_argument("--leader-can", default="can_leader")
    ap.add_argument("--follower-gripper", default="linear_4310",
                    choices=["crank_4310", "linear_3507", "linear_4310", "no_gripper"])
    ap.add_argument("--leader-gripper", default="yam_teaching_handle",
                    choices=["yam_teaching_handle", "crank_4310", "linear_3507", "linear_4310", "no_gripper"])
    ap.add_argument("--gcomp", default="1.0")
    ap.add_argument("--k-trans", default="120,120,120")
    ap.add_argument("--k-rot", default="15,15,15")
    ap.add_argument("--d-trans", default="0,0,0")
    ap.add_argument("--d-rot", default="0,0,0")
    ap.add_argument("--kd", default="3,4.5,4.5,3,2,2", help="follower joint damping floor (<=5)")
    ap.add_argument("--tau-clip", default=None, help="per-joint impedance clamp (6 vals); default = default_tau_clip()")
    ap.add_argument("--mu-c", default=None, help="per-joint Coulomb friction (6 vals); default from factr_friction")
    ap.add_argument("--mu-scale", type=float, default=0.5, help="scale applied to mu_c")
    ap.add_argument("--v-eps", type=float, default=None,
                    help="friction-comp velocity scale (higher = gentler slope, more stable vs limit cycle; default 0.05)")
    ap.add_argument("--fric-ramp", type=float, default=2.0, help="seconds to fade friction comp in after engage")
    ap.add_argument("--no-friction", action="store_true", help="disable friction comp (A/B vs clamp only)")
    ap.add_argument("--workspace", type=float, default=0.25, help="half-size of x_des box around engage (m)")
    ap.add_argument("--max-vel", type=float, default=3.0, help="guard: follower joint vel rad/s")
    ap.add_argument("--max-track", type=float, default=0.3, help="guard: tracking error (m) before abort")
    ap.add_argument("--rate", type=float, default=4.0, help="console print rate (Hz)")
    ap.add_argument("--log", default=os.path.join(_HERE, "factr_last_run.txt"),
                    help="path for the run summary (.txt)")
    ap.add_argument("--log-rate", type=float, default=30.0, help="sample rate (Hz) for the run log/summary stats")
    args = ap.parse_args()

    fol_grip = GripperType(args.follower_gripper)
    xml_path = fol_grip.get_xml_path()
    if args.selftest:
        return selftest(xml_path)

    lead_grip = GripperType(args.leader_gripper)
    K = np.concatenate([parse_vec(args.k_trans, 3, "k-trans"), parse_vec(args.k_rot, 3, "k-rot")])
    D = np.concatenate([parse_vec(args.d_trans, 3, "d-trans"), parse_vec(args.d_rot, 3, "d-rot")])
    arm_kd = parse_vec(args.kd, ARM_DOF, "kd")
    tau_clip_vec = default_tau_clip() if args.tau_clip is None else parse_vec(args.tau_clip, ARM_DOF, "tau-clip")
    fp = default_yam_params()
    if args.mu_c is not None:
        fp.mu_c = parse_vec(args.mu_c, ARM_DOF, "mu-c")
    fp.mu_c = fp.mu_c * args.mu_scale
    if args.v_eps is not None:
        fp.v_eps = np.full(ARM_DOF, args.v_eps)
    fric_on = not args.no_friction

    kin = MjKin(xml_path)

    print("=" * 72)
    print("YAM leader-follower Cartesian impedance + FACTR friction comp")
    print(f"  follower: {args.follower_can} ({args.follower_gripper})   leader: {args.leader_can} ({args.leader_gripper})")
    print(f"  K={K}   tau_clip={tau_clip_vec}")
    print(f"  friction {'ON' if fric_on else 'OFF'}  mu_c(eff)={np.round(fp.mu_c,2)}  kd={arm_kd}  max_vel={args.max_vel}")
    print(f"  run summary -> {args.log}")
    print("=" * 72)
    print(">>> Keep a hand near BOTH arms. Lead GENTLY first. Ctrl-C re-stiffens follower. <<<")
    input("Press Enter to connect to both arms...")

    from i2rt.robots.get_robot import get_yam_robot

    print(f"connecting FOLLOWER on {args.follower_can} (stiff hold)...")
    follower = get_yam_robot(channel=args.follower_can, gripper_type=fol_grip, zero_gravity_mode=False)
    print(f"connecting LEADER on {args.leader_can} (gravity float, backdrivable)...")
    leader = get_yam_robot(channel=args.leader_can, gripper_type=lead_grip, zero_gravity_mode=True)

    info = follower.get_robot_info()
    n_motors = follower.num_dofs()
    gripper_index = info.get("gripper_index", None)
    stock_kp = np.asarray(info["kp"], float).copy()
    stock_kd = np.asarray(info["kd"], float).copy()

    factor = np.ones(n_motors)
    gparts = [float(x) for x in args.gcomp.split(",") if x.strip() != ""]
    factor[:ARM_DOF] = gparts[0] if len(gparts) == 1 else np.array(gparts)[:ARM_DOF]
    follower.gravity_comp_factor = factor

    def q_of(robot):
        return np.asarray(robot.get_observations()["joint_pos"], float)[:ARM_DOF]

    def restiffen():
        try:
            follower.update_kp_kd(stock_kp, stock_kd)
            follower.command_joint_pos(follower.get_joint_pos())
        except Exception as e:  # noqa: BLE001
            print(f"  (restiffen warning: {e})")

    def on_sigint(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_sigint)

    print("\n[Phase A] Follower stiff, leader floating. Move the leader where you like.")
    t_end = time.perf_counter() + 5.0
    while time.perf_counter() < t_end:
        xf, _ = kin.fk(q_of(follower)); xl, _ = kin.fk(q_of(leader))
        print(f"  follower EE {np.round(xf,3)}   leader EE {np.round(xl,3)}   gap {np.linalg.norm(xf-xl)*100:5.1f} cm", end="\r")
        time.sleep(0.1)
    print()

    try:
        ans = input("Type 'engage' + Enter to start leader-follower tracking: ").strip()
    except KeyboardInterrupt:
        ans = ""
    if ans.lower() != "engage":
        print("Not engaging. Re-stiffening."); restiffen(); time.sleep(0.3); return 0

    qf0, ql0 = q_of(follower), q_of(leader)
    xf0_pos, Rf0 = kin.fk(qf0)
    xl0_pos, Rl0 = kin.fk(ql0)
    pos_offset = xf0_pos - xl0_pos
    R_offset = Rl0.T @ Rf0
    box_lo = xf0_pos - args.workspace
    box_hi = xf0_pos + args.workspace

    kp_full = np.zeros(n_motors)
    kd_full = np.zeros(n_motors); kd_full[:ARM_DOF] = arm_kd
    if gripper_index is not None:
        kp_full[gripper_index] = stock_kp[gripper_index]
        kd_full[gripper_index] = stock_kd[gripper_index]

    tau_total_clip = tau_clip_vec + fp.fric_max
    cfg = {
        "K_trans": np.round(K[:3], 1).tolist(), "K_rot": np.round(K[3:], 1).tolist(),
        "D": np.round(D, 1).tolist(), "tau_clip": np.round(tau_clip_vec, 1).tolist(),
        "friction": "ON" if fric_on else "OFF", "mu_c(eff)": np.round(fp.mu_c, 2).tolist(),
        "fric_max": np.round(fp.fric_max, 1).tolist(), "kd": np.round(arm_kd, 1).tolist(),
        "v_eps": np.round(fp.v_eps, 3).tolist(),
        "max_vel": args.max_vel, "workspace_m": args.workspace, "mu_scale": args.mu_scale,
    }
    rec = {k: [] for k in ("t", "ql", "qf", "xl", "xf", "perr", "rerr", "ti", "tf", "qd")}
    ended = "normal"

    print(f"\n[Phase B] TRACKING (friction {'ON' if fric_on else 'OFF'}). Move the LEADER; follower follows.")
    print(f"  follower start EE {np.round(xf0_pos,3)}  (x_des clamped +-{args.workspace} m). Ctrl-C to stop.")
    t0 = time.perf_counter()
    dt_print = 1.0 / max(args.rate, 0.5)
    dt_log = 1.0 / max(args.log_rate, 1.0)
    next_print = t0
    next_logrec = t0
    try:
        while True:
            ql = q_of(leader)
            xl_pos, Rl = kin.fk(ql)
            x_des_pos = np.clip(xl_pos + pos_offset, box_lo, box_hi)
            R_des = Rl @ R_offset

            qf = q_of(follower)
            qdf = np.asarray(follower.get_observations()["joint_vel"], float)[:ARM_DOF]
            full_pos = np.asarray(follower.get_joint_pos(), float)
            xf_pos, Rf = kin.fk(qf)
            J = kin.jacobian(qf)
            xdot = J @ qdf
            dx = np.concatenate([x_des_pos - xf_pos, Rotation.from_matrix(R_des @ Rf.T).as_rotvec()])
            F = K * dx - D * xdot

            tau_imp = np.clip(J.T @ F, -tau_clip_vec, tau_clip_vec)
            if fric_on:
                enable = min(1.0, (time.perf_counter() - t0) / max(args.fric_ramp, 1e-3))
                tau_fric = friction_torque(qdf, tau_imp, fp, enable)
            else:
                tau_fric = np.zeros(ARM_DOF)
            tau_arm = np.clip(tau_imp + tau_fric, -tau_total_clip, tau_total_clip)

            track = np.linalg.norm(dx[:3])
            if track > args.max_track or np.any(np.abs(qdf) > args.max_vel):
                print(f"\n!! GUARD: track_err={track*100:.1f}cm  max_vel={np.abs(qdf).max():.2f} — re-stiffening.")
                ended = "guard"
                restiffen(); break

            torque_full = np.zeros(n_motors)
            torque_full[:ARM_DOF] = tau_arm
            follower.command_joint_torque(torque_full, kp=kp_full, kd=kd_full, pos=full_pos)

            now = time.perf_counter()
            if now >= next_logrec:
                rec["t"].append(now - t0)
                rec["ql"].append(ql.copy()); rec["qf"].append(qf.copy())
                rec["xl"].append(xl_pos.copy()); rec["xf"].append(xf_pos.copy())
                rec["perr"].append(dx[:3].copy()); rec["rerr"].append(dx[3:].copy())
                rec["ti"].append(tau_imp.copy()); rec["tf"].append(tau_fric.copy()); rec["qd"].append(qdf.copy())
                next_logrec = now + dt_log
            if now >= next_print:
                print("-" * 72)
                print(f"leader EE {np.round(xl_pos,3)}  track_err {track*100:5.1f} cm")
                print(f"tau_imp  {np.round(tau_imp,2)}   tau_fric {np.round(tau_fric,2)}")
                next_print = now + dt_print
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nCtrl-C — re-stiffening follower.")
        ended = "ctrl-c"
        restiffen()

    write_summary(args.log, cfg, rec, tau_clip_vec, fric_on, ended)
    print("\nDone. Follower in stiff hold. SUPPORT BOTH ARMS — they relax when this exits.")
    try:
        input("Press Enter to exit (keep holding the arms)...")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
