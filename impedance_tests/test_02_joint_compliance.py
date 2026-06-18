#!/usr/bin/env python3
"""Test 02 — Joint-space compliance (joint-space impedance) on a single YAM arm.

Builds directly on Test 01. Test 01 proved gravity compensation holds the arm
weightless at gravity_comp_factor=1.0 — but with kp=0 the arm has NO position
memory, so a nudge makes it drift (that's the "it sagged when I moved it"
behavior). This test adds the missing piece: a STIFFNESS term.

  joint-space impedance:  tau = kp*(q_target - q) + kd*(0 - qdot) + g(q)
                                \_________ spring _________/          \grav/

With a modest kp the arm HOLDS q_target and behaves like a soft spring: push it
away and it returns. This is the simplest real impedance controller, and it's a
safe checkpoint before we add the Cartesian / raw-torque machinery (adding
stiffness only makes the arm MORE stable).

This uses ONLY the public i2rt API (update_kp_kd + command_joint_pos); no source
changes. Gravity is handled by i2rt's built-in comp (factor 1.0, validated in
Test 01). Defaults for kp/kd come from the Trossen joint-space impedance design.

SAFETY
  - Brings up in STIFF HOLD; you opt into the compliant mode by typing 'engage'.
  - Runaway guard re-stiffens if a joint exceeds --max-vel or drifts past
    --max-drift from the engage pose.
  - The gripper always keeps its stock position gains (never goes compliant).
  - On exit the arm re-stiffens; the ~400 ms motor watchdog relaxes it once the
    process dies, so SUPPORT THE ARM before the final exit.

USAGE (in the env with i2rt, e.g. miniforge 'lerobot'):
  python test_02_joint_compliance.py --channel can_follower --gripper linear_4310
  # softer spring:
  python test_02_joint_compliance.py --channel can_follower --kp 6,6,6,3,2,2
  # stiffer spring:
  python test_02_joint_compliance.py --channel can_follower --kp 25,25,25,12,8,8
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

# Use the i2rt from THIS repo (the impedance-control-testing branch), not the
# installed site-packages copy, so branch edits take effect.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from i2rt.robots.get_robot import get_yam_robot  # noqa: E402
from i2rt.robots.utils import GripperType  # noqa: E402
from i2rt.utils.mujoco_utils import MuJoCoKDL  # noqa: E402

ARM_DOF = 6


def parse_vec6(s: str, name: str) -> np.ndarray:
    parts = [float(x) for x in s.split(",") if x.strip() != ""]
    if len(parts) == 1:
        return np.full(ARM_DOF, parts[0])
    if len(parts) == ARM_DOF:
        return np.array(parts)
    raise ValueError(f"--{name} expects 1 or {ARM_DOF} comma-separated values, got {len(parts)}")


def fmt_row(label: str, vals: np.ndarray) -> str:
    return label.ljust(16) + " ".join(f"{v:+7.3f}" for v in vals)


def main() -> int:
    ap = argparse.ArgumentParser(description="YAM joint-space compliance (impedance) test")
    ap.add_argument("--channel", default="can_follower", help="CAN interface the follower is on")
    ap.add_argument("--gripper", default="linear_4310",
                    choices=["crank_4310", "linear_3507", "linear_4310", "no_gripper"])
    ap.add_argument("--gcomp", default="1.0", help="gravity_comp_factor (validated = 1.0)")
    ap.add_argument("--kp", default="12,12,12,6,4,4", help="arm joint stiffness (N*m/rad)")
    ap.add_argument("--kd", default="2,2,2,1,0.6,0.6", help="arm joint damping")
    ap.add_argument("--max-vel", type=float, default=2.0, help="runaway guard: max |arm vel| rad/s")
    ap.add_argument("--max-drift", type=float, default=0.7, help="runaway guard: max |drift| rad")
    ap.add_argument("--rate", type=float, default=4.0, help="console log rate (Hz)")
    args = ap.parse_args()

    gripper_type = GripperType(args.gripper)
    xml_path = gripper_type.get_xml_path()
    arm_kp = parse_vec6(args.kp, "kp")
    arm_kd = parse_vec6(args.kd, "kd")

    print("=" * 72)
    print("YAM joint-space compliance (joint-space impedance) test")
    print(f"  channel : {args.channel}   gripper: {args.gripper}")
    print(f"  arm kp  : {arm_kp}")
    print(f"  arm kd  : {arm_kd}")
    print("  bringup : STIFF HOLD — arm will NOT move on start")
    print("=" * 72)
    print(">>> Keep a hand near the arm. Ctrl-C re-stiffens and exits. <<<")
    input("Press Enter to connect to the arm...")

    robot = get_yam_robot(channel=args.channel, gripper_type=gripper_type, zero_gravity_mode=False)
    info = robot.get_robot_info()
    n_motors = robot.num_dofs()

    factor = np.ones(n_motors)
    gparts = [float(x) for x in args.gcomp.split(",") if x.strip() != ""]
    factor[:ARM_DOF] = gparts[0] if len(gparts) == 1 else np.array(gparts)[:ARM_DOF]
    robot.gravity_comp_factor = factor

    stock_kp = np.asarray(info["kp"], dtype=float).copy()
    stock_kd = np.asarray(info["kd"], dtype=float).copy()
    kdl = MuJoCoKDL(xml_path)

    print(f"\nrobot up. n_motors={n_motors}  gravity_comp_factor={np.round(factor, 2)}")

    def restiffen():
        try:
            robot.update_kp_kd(stock_kp, stock_kd)
            robot.command_joint_pos(robot.get_joint_pos())
        except Exception as e:  # noqa: BLE001
            print(f"  (restiffen warning: {e})")

    def on_sigint(signum, frame):  # noqa: ARG001
        print("\nCtrl-C — re-stiffening and exiting. SUPPORT THE ARM.")
        restiffen()
        time.sleep(0.3)
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)
    dt = 1.0 / max(args.rate, 0.5)

    def log_once(target_arm):
        obs = robot.get_observations()
        q = np.asarray(obs["joint_pos"], dtype=float)[:ARM_DOF]
        vel = np.asarray(obs["joint_vel"], dtype=float)[:ARM_DOF]
        eff = np.asarray(obs["joint_eff"], dtype=float)[:ARM_DOF]
        g = kdl.compute_inverse_dynamics(q, np.zeros(ARM_DOF), np.zeros(ARM_DOF))
        print("-" * 72)
        print(fmt_row("pos (rad)", q))
        print(fmt_row("target - pos", target_arm - q))     # spring deflection
        print(fmt_row("vel (rad/s)", vel))
        print(fmt_row("eff measured", eff))
        print(fmt_row("g(q) model", g))

    # ---- Phase A: stiff hold ----
    print("\n[Phase A] Stiff hold. Press Enter-prompt below to go compliant.")
    t_end = time.perf_counter() + 4.0
    while time.perf_counter() < t_end:
        time.sleep(dt)

    ans = input("Type 'engage' + Enter to make the arm a soft spring at its current pose: ").strip()
    if ans.lower() != "engage":
        print("Not engaging. Re-stiffening and exiting.")
        restiffen()
        time.sleep(0.3)
        return 0

    # ---- Phase B: joint-space compliant hold ----
    target_full = np.asarray(robot.get_joint_pos(), dtype=float)
    target_arm = target_full[:ARM_DOF]
    kp_full = stock_kp.copy(); kp_full[:ARM_DOF] = arm_kp
    kd_full = stock_kd.copy(); kd_full[:ARM_DOF] = arm_kd
    robot.update_kp_kd(kp_full, kd_full)
    robot.command_joint_pos(target_full)  # holds this pose with the soft gains + gravity comp

    print(f"\n[Phase B] COMPLIANT. Target pose locked at {np.round(target_arm, 3)}")
    print("  >>> Gently push the arm away and let go — it should spring back toward the target. <<<")
    print("  >>> Higher --kp = stiffer; lower = softer. Ctrl-C when done (support the arm). <<<")

    next_log = time.perf_counter()
    try:
        while True:
            full_pos = np.asarray(robot.get_joint_pos(), dtype=float)
            vel_full = np.asarray(robot.get_observations()["joint_vel"], dtype=float)
            arm_pos = full_pos[:ARM_DOF]
            arm_vel = vel_full[:ARM_DOF]
            drift = np.abs(arm_pos - target_arm)
            if np.any(np.abs(arm_vel) > args.max_vel) or np.any(drift > args.max_drift):
                bad = int(np.argmax(np.maximum(np.abs(arm_vel) / args.max_vel, drift / args.max_drift)))
                print(f"\n!! RUNAWAY GUARD tripped on joint {bad} "
                      f"(vel={arm_vel[bad]:+.3f}, drift={drift[bad]:.3f}) — re-stiffening.")
                restiffen()
                break
            now = time.perf_counter()
            if now >= next_log:
                log_once(target_arm)
                next_log = now + dt
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nCtrl-C in compliant mode — re-stiffening.")
        restiffen()

    print("\nDone. Arm is in stiff hold. SUPPORT THE ARM — it relaxes when this process exits.")
    input("Press Enter to exit (keep holding the arm)...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
