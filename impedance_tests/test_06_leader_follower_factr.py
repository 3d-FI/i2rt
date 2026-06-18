#!/usr/bin/env python3
"""Test 06 — Leader-follower Cartesian impedance WITH FACTR friction compensation.

Same as test_04 (leader's EE pose drives x_des; follower tracks it with Cartesian
impedance), but adds:
  - per-joint torque budget (real motor limits: DM4340 base/shoulder/elbow up to ~28 N*m,
    DM4310 wrist ~10) instead of the flat 4 N*m clamp, and
  - FACTR-style friction compensation (factr_friction.py) so the heavy shoulder/elbow
    break loose from stiction and actually follow the leader.

Control (follower, per cycle):
    tau_imp  = clip(J^T F, -tau_clip_vec, +tau_clip_vec)        # per-joint impedance clamp
    enable   = ramp 0->1 over --fric-ramp seconds after engage
    tau_fric = friction_torque(qdot, tau_imp, params, enable)    # breakaway + viscous comp
    tau_arm  = clip(tau_imp + tau_fric, -(tau_clip+fric_max), +(tau_clip+fric_max))
    command_joint_torque([tau_arm, gripper=0], kp=0 arm, kd=joint_damping, pos=hold gripper)
  (gravity g(q) is added on top by i2rt; friction comp never touches the gripper.)

SAFETY: follower up STIFF, opt in with 'engage'; leader floats (read-only); x_des clamped
to a workspace box; per-joint torque budget; friction enable ramps in; runaway guard on
follower velocity and tracking error (also catches friction self-motion). Push/lead GENTLY.
Ctrl-C re-stiffens the follower. SUPPORT BOTH ARMS before exit.

OFFLINE (no hardware):  python test_06_leader_follower_factr.py --selftest

HARDWARE (after calibrating mu_c with test_05):
  python test_06_leader_follower_factr.py --follower-can can_follower --leader-can can_leader \
      --mu-c <paste from test_05> --mu-scale 0.5
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

# friction module is in the same dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factr_friction import (  # noqa: E402
    FrictionParams, default_yam_params, default_tau_clip, friction_torque,
    check_budget, friction_curve_demo, GRAVITY_MAX,
)

from i2rt.robots.utils import GripperType  # noqa: E402

ARM_DOF = 6


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
    """Offline: Jacobian finite-diff check + friction curve. No hardware."""
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
    ap.add_argument("--k-rot", default="6,6,6")
    ap.add_argument("--d-trans", default="0,0,0")
    ap.add_argument("--d-rot", default="0,0,0")
    ap.add_argument("--kd", default="3,3,3,2,1,1", help="follower joint damping floor (<=5)")
    ap.add_argument("--tau-clip", default=None, help="per-joint impedance clamp (6 vals); default = factr_friction.default_tau_clip()")
    ap.add_argument("--mu-c", default=None, help="per-joint Coulomb friction (6 vals); default from factr_friction")
    ap.add_argument("--mu-scale", type=float, default=0.5, help="scale applied to mu_c (stay below true breakaway)")
    ap.add_argument("--fric-ramp", type=float, default=2.0, help="seconds to fade friction comp in after engage")
    ap.add_argument("--no-friction", action="store_true", help="disable friction comp (A/B vs test_04 + per-joint clamp)")
    ap.add_argument("--workspace", type=float, default=0.25, help="half-size of x_des box around engage (m)")
    ap.add_argument("--max-vel", type=float, default=2.0, help="guard: follower joint vel rad/s")
    ap.add_argument("--max-track", type=float, default=0.3, help="guard: tracking error (m) before abort")
    ap.add_argument("--rate", type=float, default=4.0)
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

    # Friction params: defaults, optionally override mu_c with calibrated values, then scale.
    fp = default_yam_params()
    if args.mu_c is not None:
        fp.mu_c = parse_vec(args.mu_c, ARM_DOF, "mu-c")
    fp.mu_c = fp.mu_c * args.mu_scale
    fric_on = not args.no_friction

    kin = MjKin(xml_path)

    print("=" * 72)
    print("YAM leader-follower Cartesian impedance + FACTR friction comp")
    print(f"  follower: {args.follower_can} ({args.follower_gripper})   leader: {args.leader_can} ({args.leader_gripper})")
    print(f"  K={K}")
    print(f"  tau_clip(per-joint)={tau_clip_vec}")
    print(f"  friction {'ON' if fric_on else 'OFF'}  mu_c(scaled)={np.round(fp.mu_c,2)}  fric_max={fp.fric_max}  ramp={args.fric_ramp}s")
    print(f"  joint kd={arm_kd}  max_vel={args.max_vel}  workspace=+-{args.workspace}m")
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
        print("\nCtrl-C — re-stiffening follower. SUPPORT BOTH ARMS.")
        restiffen(); time.sleep(0.3); sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    # ---- Phase A ----
    print("\n[Phase A] Follower stiff, leader floating. Move the leader where you like.")
    t_end = time.perf_counter() + 5.0
    while time.perf_counter() < t_end:
        xf, _ = kin.fk(q_of(follower)); xl, _ = kin.fk(q_of(leader))
        print(f"  follower EE {np.round(xf,3)}   leader EE {np.round(xl,3)}   gap {np.linalg.norm(xf-xl)*100:5.1f} cm", end="\r")
        time.sleep(0.1)
    print()

    ans = input("Type 'engage' + Enter to start leader-follower tracking: ").strip()
    if ans.lower() != "engage":
        print("Not engaging. Re-stiffening."); restiffen(); time.sleep(0.3); return 0

    # ---- Clutch ----
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
    t_engage = time.perf_counter()
    print(f"\n[Phase B] TRACKING (friction {'ON' if fric_on else 'OFF'}). Move the LEADER; follower follows.")
    print(f"  follower start EE {np.round(xf0_pos,3)}  (x_des clamped +-{args.workspace} m)")
    dt = 1.0 / max(args.rate, 0.5)
    next_log = time.perf_counter()
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
                enable = min(1.0, (time.perf_counter() - t_engage) / max(args.fric_ramp, 1e-3))
                tau_fric = friction_torque(qdf, tau_imp, fp, enable)
            else:
                tau_fric = np.zeros(ARM_DOF)
            tau_arm = np.clip(tau_imp + tau_fric, -tau_total_clip, tau_total_clip)

            track = np.linalg.norm(dx[:3])
            if track > args.max_track or np.any(np.abs(qdf) > args.max_vel):
                print(f"\n!! GUARD: track_err={track*100:.1f}cm  max_vel={np.abs(qdf).max():.2f} — re-stiffening.")
                restiffen(); break

            torque_full = np.zeros(n_motors)
            torque_full[:ARM_DOF] = tau_arm
            follower.command_joint_torque(torque_full, kp=kp_full, kd=kd_full, pos=full_pos)

            now = time.perf_counter()
            if now >= next_log:
                viol = check_budget(tau_imp, tau_fric, GRAVITY_MAX, arm_kd, qdf)
                print("-" * 72)
                print(f"leader EE {np.round(xl_pos,3)}  x_des {np.round(x_des_pos,3)}  track_err {track*100:5.1f} cm")
                print(f"follow EE {np.round(xf_pos,3)}")
                print(f"tau_imp  {np.round(tau_imp,2)}")
                print(f"tau_fric {np.round(tau_fric,2)}")
                if viol:
                    print("  !! budget: " + "; ".join(viol))
                next_log = now + dt
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nCtrl-C — re-stiffening follower.")
        restiffen()

    print("\nDone. Follower in stiff hold. SUPPORT BOTH ARMS — they relax when this exits.")
    input("Press Enter to exit (keep holding the arms)...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
