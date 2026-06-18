#!/usr/bin/env python3
"""Test 05 — Per-joint breakaway-friction calibration for the YAM arm.

Measures the static/breakaway friction tau_s of each arm joint so the FACTR friction
compensation (factr_friction.py / test_06) can be tuned. For one joint at a time:

  - The whole arm is held STIFF and gravity-compensated.
  - We FREE a single joint (kp=0, light kd), so it is gravity-balanced and held only by
    its own friction.
  - We slowly ramp a feedforward torque on that joint until it starts to move
    (|qdot| > threshold) -> that torque is the breakaway friction in that direction.
  - We do both directions and average (the +/- average cancels gravity-model bias).

Output: a length-6 mu_c array. Use ~0.5x of it as the friction-comp mu_c in test_06.

SAFETY: one joint at a time, every other joint stays rigidly position-held; slow ramp;
small per-joint probe cap; immediate re-stiffen the instant motion is detected; abort on
excess velocity/drift; gripper position-held; SIGINT re-stiffens. Keep a hand near the arm.

USAGE:
  python test_05_friction_calibrate.py --channel can_follower --joint all
  python test_05_friction_calibrate.py --channel can_follower --joint 1   # one joint
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

from i2rt.robots.utils import GripperType  # noqa: E402

ARM_DOF = 6
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist-pitch", "wrist-yaw", "wrist-roll"]
# DM4340 (0,1,2) tolerate a larger probe than DM4310 (3,4,5).
PROBE_CAP = np.array([6.0, 6.0, 6.0, 4.0, 4.0, 4.0])


def main() -> int:
    ap = argparse.ArgumentParser(description="YAM per-joint breakaway-friction calibration")
    ap.add_argument("--channel", default="can_follower")
    ap.add_argument("--gripper", default="linear_4310",
                    choices=["crank_4310", "linear_3507", "linear_4310", "no_gripper"])
    ap.add_argument("--joint", default="all", help="'all' or a joint index 0-5")
    ap.add_argument("--gcomp", default="1.0", help="gravity_comp_factor (validated 1.0)")
    ap.add_argument("--ramp-rate", type=float, default=0.5, help="torque ramp rate (N*m/s)")
    ap.add_argument("--vel-thresh", type=float, default=0.05, help="breakaway velocity (rad/s)")
    ap.add_argument("--vel-abort", type=float, default=0.4, help="abort velocity (rad/s)")
    ap.add_argument("--drift-abort", type=float, default=0.25, help="abort joint drift (rad)")
    ap.add_argument("--free-kd", type=float, default=1.0, help="kd left on the freed joint")
    args = ap.parse_args()

    if args.joint == "all":
        joints = list(range(ARM_DOF))
    else:
        joints = [int(args.joint)]
        if not 0 <= joints[0] < ARM_DOF:
            print(f"--joint must be 0-{ARM_DOF-1} or 'all'"); return 2

    gripper_type = GripperType(args.gripper)

    print("=" * 72)
    print("YAM breakaway-friction calibration")
    print(f"  channel {args.channel}   gripper {args.gripper}   joints {joints}")
    print("  Arm holds STIFF; one joint is freed and ramped until it breaks loose.")
    print("=" * 72)
    print(">>> Keep a hand near the arm. Ctrl-C re-stiffens and exits. <<<")
    input("Press Enter to connect...")

    from i2rt.robots.get_robot import get_yam_robot

    # Bring up FLOATING (zero_gravity_mode=True -> kp=kd=0 + gravity comp) so the arm is
    # backdrivable and you can hand-position it before we lock and calibrate.
    robot = get_yam_robot(channel=args.channel, gripper_type=gripper_type, zero_gravity_mode=True)
    info = robot.get_robot_info()
    n_motors = robot.num_dofs()
    stock_kp = np.asarray(info["kp"], float).copy()
    stock_kd = np.asarray(info["kd"], float).copy()

    factor = np.ones(n_motors)
    gparts = [float(x) for x in args.gcomp.split(",") if x.strip() != ""]
    factor[:ARM_DOF] = gparts[0] if len(gparts) == 1 else np.array(gparts)[:ARM_DOF]
    robot.gravity_comp_factor = factor

    def restiffen():
        robot.update_kp_kd(stock_kp, stock_kd)
        robot.command_joint_pos(robot.get_joint_pos())

    def on_sigint(signum, frame):  # noqa: ARG001
        print("\nCtrl-C — re-stiffening. SUPPORT THE ARM.")
        try:
            restiffen()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3); sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    # Position the arm while it FLOATS, then lock it stiff for calibration.
    print("\nArm is FLOATING (backdrivable). Move it by hand to a good calibration pose:")
    print("  - bend the elbow well OFF its lower limit (0 rad); keep all joints mid-range,")
    print("    away from limits, so each joint can move freely both directions.")
    input("Position the arm, keep a hand on it, then press Enter to LOCK and start...")
    restiffen()
    time.sleep(0.6)
    print(f"locked at {np.round(np.asarray(robot.get_joint_pos(), float)[:ARM_DOF], 3)}")

    def vel(j):
        return float(np.asarray(robot.get_observations()["joint_vel"], float)[j])

    def pos(j):
        return float(np.asarray(robot.get_observations()["joint_pos"], float)[j])

    def ramp_one(j: int, sign: int) -> float | None:
        """Ramp a feedforward torque on joint j (sign +/-1) until breakaway. Returns the
        breakaway torque magnitude, or None if the probe cap was reached with no motion."""
        restiffen()
        time.sleep(0.8)                      # settle
        hold_full = np.asarray(robot.get_joint_pos(), float)
        hold_j = hold_full[j]
        kp_free = stock_kp.copy(); kp_free[j] = 0.0
        kd_free = stock_kd.copy(); kd_free[j] = args.free_kd
        cap = PROBE_CAP[j]
        dt = 0.02
        tau = 0.0
        consec = 0
        while True:
            tau += sign * args.ramp_rate * dt
            torque_full = np.zeros(n_motors)
            torque_full[j] = tau
            robot.command_joint_torque(torque_full, kp=kp_free, kd=kd_free, pos=hold_full)
            time.sleep(dt)
            v = vel(j)
            drift = abs(pos(j) - hold_j)
            if abs(v) > args.vel_thresh:
                consec += 1
            else:
                consec = 0
            if consec >= 2:
                restiffen()
                return abs(tau)
            if abs(v) > args.vel_abort or drift > args.drift_abort:
                restiffen()
                return abs(tau)              # broke loose (fast) — still a valid breakaway
            if abs(tau) >= cap:
                restiffen()
                return None                  # no breakaway within probe cap
        # unreachable

    mu_c = np.full(ARM_DOF, np.nan)
    results = []
    for j in joints:
        print(f"\n--- joint {j} ({JOINT_NAMES[j]}) ---  probe cap {PROBE_CAP[j]:.1f} N*m")
        input(f"Arm is locked at the positioned pose. Hand near it, press Enter to calibrate joint {j}...")
        b_plus = ramp_one(j, +1)
        print(f"  +dir breakaway: {('%.3f N*m' % b_plus) if b_plus is not None else 'none up to cap'}")
        time.sleep(0.5)
        b_minus = ramp_one(j, -1)
        print(f"  -dir breakaway: {('%.3f N*m' % b_minus) if b_minus is not None else 'none up to cap'}")
        vals = [b for b in (b_plus, b_minus) if b is not None]
        if vals:
            mu_c[j] = float(np.mean(vals))
        results.append((j, b_plus, b_minus, mu_c[j]))

    print("\n" + "=" * 72)
    print("CALIBRATION RESULTS")
    print(f"{'joint':>5} {'name':>12} {'+brk':>8} {'-brk':>8} {'mu_c=tau_s':>11}")
    for j, bp, bm, mc in results:
        sp = f"{bp:.3f}" if bp is not None else "   -"
        sm = f"{bm:.3f}" if bm is not None else "   -"
        sc = f"{mc:.3f}" if not np.isnan(mc) else "  nan"
        print(f"{j:>5} {JOINT_NAMES[j]:>12} {sp:>8} {sm:>8} {sc:>11}")

    full = np.where(np.isnan(mu_c), 0.0, mu_c)
    print("\nMeasured breakaway (tau_s) per joint:")
    print("  " + ",".join(f"{v:.2f}" for v in full))
    print("\nSuggested friction-comp mu_c for test_06 (0.5x, stays below breakaway):")
    print("  --mu-c " + ",".join(f"{0.5 * v:.2f}" for v in full))
    print("\n(re-run at a couple of arm poses and average if you want a robust number.)")
    print("Re-stiffening. SUPPORT THE ARM, then press Enter to exit.")
    restiffen()
    input("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
