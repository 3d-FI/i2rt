#!/usr/bin/env python3
"""Test 04 — Leader-follower Cartesian impedance (two YAM arms).

The LEADER arm (teaching handle, backdrivable / gravity-floated) is moved by hand.
Its end-effector pose drives the target x_des; the FOLLOWER arm tracks x_des with
the Cartesian impedance controller from Test 03 (compliant, directional). So you
move the leader's gripper and the follower's gripper follows compliantly.

    x_leader = FK(q_leader)                       # leader EE pose (read-only)
    x_des    = x_leader + clutch_offset           # offset captured at engage (no jump)
    F        = K*(x_des - x_follower) - D*x_dot    # Cartesian wrench on follower
    tau      = J_follower^T @ F  -> command_joint_torque(tau)   # gravity added by i2rt

CLUTCH: at engage we capture offset = (follower pose) - (leader pose), so the
follower does NOT lunge to match the leader; it starts exactly where it is and
tracks the leader's *motion* from there.

SAFETY
  - Follower brought up STIFF; opt in with 'engage'. Leader floats (gravity comp,
    kp=kd=0) the whole time and is never commanded.
  - x_des is clamped to a workspace box around the engage pose (leader can't drive
    the follower out of reach).
  - Impedance torque clamped (|J^T F| <= --tau-clip); joint-damping floor; runaway
    guard on follower joint velocity and on tracking error.
  - Gripper held in position mode. Push/lead GENTLY first. Ctrl-C re-stiffens the
    follower. SUPPORT BOTH ARMS before the process exits.

Needs BOTH CAN buses up and BOTH arms powered:
    sudo ip link set can_follower up type can bitrate 1000000
    sudo ip link set can_leader   up type can bitrate 1000000

USAGE:
    python test_04_leader_follower_cartesian.py \
        --follower-can can_follower --leader-can can_leader
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

from i2rt.robots.utils import GripperType  # noqa: E402

ARM_DOF = 6


class MjKin:
    """FK + 6x6 site Jacobian for the YAM arm (shared by leader & follower; arm
    kinematics are identical, so one model serves both)."""

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
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return np.vstack([jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]])


def parse_vec(s, n, name):
    parts = [float(x) for x in s.split(",") if x.strip() != ""]
    if len(parts) == 1:
        return np.full(n, parts[0])
    if len(parts) == n:
        return np.array(parts)
    raise ValueError(f"--{name} expects 1 or {n} values, got {len(parts)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="YAM leader-follower Cartesian impedance")
    ap.add_argument("--follower-can", default="can_follower")
    ap.add_argument("--leader-can", default="can_leader")
    ap.add_argument("--follower-gripper", default="linear_4310",
                    choices=["crank_4310", "linear_3507", "linear_4310", "no_gripper"])
    ap.add_argument("--leader-gripper", default="yam_teaching_handle",
                    choices=["yam_teaching_handle", "crank_4310", "linear_3507", "linear_4310", "no_gripper"])
    ap.add_argument("--gcomp", default="1.0", help="follower gravity_comp_factor (validated 1.0)")
    ap.add_argument("--k-trans", default="120,120,120")
    ap.add_argument("--k-rot", default="6,6,6")
    ap.add_argument("--d-trans", default="0,0,0")
    ap.add_argument("--d-rot", default="0,0,0")
    ap.add_argument("--kd", default="3,3,3,2,1,1", help="follower joint damping floor (<=5)")
    ap.add_argument("--tau-clip", type=float, default=4.0)
    ap.add_argument("--workspace", type=float, default=0.25, help="half-size of x_des box around engage (m)")
    ap.add_argument("--max-vel", type=float, default=2.5, help="guard: follower joint vel rad/s")
    ap.add_argument("--max-track", type=float, default=0.3, help="guard: tracking error (m) before abort")
    ap.add_argument("--rate", type=float, default=4.0)
    args = ap.parse_args()

    fol_grip = GripperType(args.follower_gripper)
    lead_grip = GripperType(args.leader_gripper)
    xml_path = fol_grip.get_xml_path()
    K = np.concatenate([parse_vec(args.k_trans, 3, "k-trans"), parse_vec(args.k_rot, 3, "k-rot")])
    D = np.concatenate([parse_vec(args.d_trans, 3, "d-trans"), parse_vec(args.d_rot, 3, "d-rot")])
    arm_kd = parse_vec(args.kd, ARM_DOF, "kd")
    kin = MjKin(xml_path)

    print("=" * 72)
    print("YAM leader-follower Cartesian impedance")
    print(f"  follower: {args.follower_can} ({args.follower_gripper})   leader: {args.leader_can} ({args.leader_gripper})")
    print(f"  K={K}  kd={arm_kd}  tau_clip={args.tau_clip}  workspace=+-{args.workspace}m")
    print("  follower brings up STIFF; leader floats (move it by hand).")
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

    # ---- Phase A: show the gap between leader and follower so the clutch makes sense ----
    print("\n[Phase A] Follower stiff, leader floating. Move the leader where you like.")
    t_end = time.perf_counter() + 5.0
    while time.perf_counter() < t_end:
        xf, _ = kin.fk(q_of(follower))
        xl, _ = kin.fk(q_of(leader))
        print(f"  follower EE {np.round(xf,3)}   leader EE {np.round(xl,3)}   gap {np.linalg.norm(xf-xl)*100:5.1f} cm", end="\r")
        time.sleep(0.1)
    print()

    ans = input("Type 'engage' + Enter to start leader-follower tracking: ").strip()
    if ans.lower() != "engage":
        print("Not engaging. Re-stiffening."); restiffen(); time.sleep(0.3); return 0

    # ---- Clutch: capture offset so the follower does NOT jump ----
    qf0, ql0 = q_of(follower), q_of(leader)
    xf0_pos, Rf0 = kin.fk(qf0)
    xl0_pos, Rl0 = kin.fk(ql0)
    pos_offset = xf0_pos - xl0_pos
    R_offset = Rl0.T @ Rf0            # so R_des(engage) = Rl0 @ R_offset = Rf0
    box_lo = xf0_pos - args.workspace
    box_hi = xf0_pos + args.workspace

    kp_full = np.zeros(n_motors)
    kd_full = np.zeros(n_motors); kd_full[:ARM_DOF] = arm_kd
    if gripper_index is not None:
        kp_full[gripper_index] = stock_kp[gripper_index]
        kd_full[gripper_index] = stock_kd[gripper_index]

    print(f"\n[Phase B] TRACKING. Move the LEADER; the follower follows compliantly.")
    print(f"  follower start EE {np.round(xf0_pos,3)}  (x_des clamped to +-{args.workspace} m around it)")
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
            tau = np.clip(J.T @ F, -args.tau_clip, args.tau_clip)

            track = np.linalg.norm(dx[:3])
            if track > args.max_track or np.any(np.abs(qdf) > args.max_vel):
                print(f"\n!! GUARD: track_err={track*100:.1f}cm  max_vel={np.abs(qdf).max():.2f} — re-stiffening.")
                restiffen(); break

            torque_full = np.zeros(n_motors)
            torque_full[:ARM_DOF] = tau
            follower.command_joint_torque(torque_full, kp=kp_full, kd=kd_full, pos=full_pos)

            now = time.perf_counter()
            if now >= next_log:
                print("-" * 72)
                print(f"leader EE {np.round(xl_pos,3)}  x_des {np.round(x_des_pos,3)}")
                print(f"follow EE {np.round(xf_pos,3)}  track_err {track*100:5.1f} cm")
                print(f"wrench F  {np.round(F,2)}   tau {np.round(tau,2)}")
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
