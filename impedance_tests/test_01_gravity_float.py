#!/usr/bin/env python3
"""Test 01 — Gravity-compensation characterization & float/hold test (single YAM arm).

This is the first bring-up test for Cartesian impedance control. Under the hood,
the whole arm is always torque-controlled (DM4310/4340 MIT mode); gravity
compensation is just a torque feedforward g(q) * gravity_comp_factor. So this
test simultaneously:

  1. Proves the torque feedforward path works end-to-end (PDF Step 1).
  2. Lets you tune gravity_comp_factor so the arm HOLDS instead of drifting up.
     (i2rt's default 1.3 over-compensates -> the arm floats UPWARD with zero
      damping. For a follower that should hold position you want ~1.0, tuned
      per joint, because the MuJoCo model excludes your real gripper/payload.)
  3. Logs measured joint effort vs. model gravity g(q) at several poses, which
     becomes the validated g(q) term the impedance controller needs.

SAFETY MODEL
  - Brings the arm up in STIFF HOLD (zero_gravity_mode=False) so it does NOT
    move on startup. You explicitly opt in to the float phase by typing 'float'.
  - In float, the arm joints get kp=0 but keep their stock kd (velocity damping),
    so a slightly-wrong factor drifts SLOWLY instead of running away.
  - Runaway guard: if any arm joint exceeds --max-vel or drifts past --max-drift
    from where you released it, the arm is immediately re-stiffened and the float
    phase aborts.
  - The gripper joint always keeps its position gains (it never floats).
  - On exit the arm is commanded back to a stiff hold. NOTE: once this process
    exits, i2rt's ~400 ms motor watchdog puts the motors into damping mode, so
    the arm will slowly settle. **Physically support the arm before exiting.**

USAGE (run in the env that has i2rt, e.g. miniforge 'lerobot'):
  python impedance_tests/test_01_gravity_float.py --channel can0 --gripper linear_4310
  # hold test (no float), just log effort vs model gravity at the current pose:
  python impedance_tests/test_01_gravity_float.py --channel can0 --measure-only
  # try a lower factor (single value broadcast to all 6 arm joints):
  python impedance_tests/test_01_gravity_float.py --channel can0 --gcomp 1.0
  # per-joint factor (6 comma-separated values, base->wrist):
  python impedance_tests/test_01_gravity_float.py --channel can0 --gcomp 1.0,1.0,1.0,1.0,0.95,0.95

This script does NOT modify any i2rt source. It only uses the public robot API.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType
from i2rt.utils.mujoco_utils import MuJoCoKDL

ARM_DOF = 6  # YAM has 6 arm joints (+ 1 gripper)


def parse_factor(s: str, n_motors: int, gripper_index: int) -> np.ndarray:
    """Turn --gcomp into a full per-motor factor array (length n_motors).

    Single value  -> broadcast to the 6 arm joints, gripper kept at 1.0.
    6 values      -> arm joints, gripper appended as 1.0.
    n_motors vals -> used verbatim.
    """
    parts = [float(x) for x in s.split(",") if x.strip() != ""]
    factor = np.ones(n_motors, dtype=float)
    if len(parts) == 1:
        factor[:ARM_DOF] = parts[0]
    elif len(parts) == ARM_DOF:
        factor[:ARM_DOF] = np.array(parts)
    elif len(parts) == n_motors:
        factor = np.array(parts)
    else:
        raise ValueError(
            f"--gcomp expects 1, {ARM_DOF}, or {n_motors} comma-separated values, got {len(parts)}"
        )
    if gripper_index is not None:
        factor[gripper_index] = 1.0  # gravity comp for the gripper joint is always 0 anyway
    return factor


def fmt_row(label: str, vals: np.ndarray) -> str:
    return label.ljust(16) + " ".join(f"{v:+7.3f}" for v in vals)


def main() -> int:
    ap = argparse.ArgumentParser(description="YAM gravity-comp characterization & float test")
    ap.add_argument("--channel", default="can0", help="CAN interface the FOLLOWER arm is on (e.g. can0)")
    ap.add_argument(
        "--gripper",
        default="linear_4310",
        choices=["crank_4310", "linear_3507", "linear_4310", "no_gripper"],
        help="Must match the physical gripper (lerobot 'v3' == linear_4310)",
    )
    ap.add_argument("--gcomp", default="1.0", help="gravity_comp_factor: 1 value, 6 values, or per-motor")
    ap.add_argument("--kd-damp", type=float, default=None,
                    help="Override arm damping kd during float (default: stock kd). Lower = more backdrivable.")
    ap.add_argument("--max-vel", type=float, default=1.5, help="Runaway guard: max |arm joint vel| rad/s")
    ap.add_argument("--max-drift", type=float, default=0.6, help="Runaway guard: max |pos drift| rad from release")
    ap.add_argument("--measure-only", action="store_true", help="Only log effort vs g(q) while holding; never float")
    ap.add_argument("--rate", type=float, default=4.0, help="Console log rate (Hz)")
    args = ap.parse_args()

    gripper_type = GripperType(args.gripper)
    xml_path = gripper_type.get_xml_path()

    print("=" * 72)
    print("YAM gravity-comp characterization / float test")
    print(f"  channel       : {args.channel}")
    print(f"  gripper        : {args.gripper}  (xml: {xml_path})")
    print("  bringup        : STIFF HOLD (zero_gravity_mode=False) — arm will NOT move on start")
    print("=" * 72)
    print(">>> Keep a hand near the arm. Press Ctrl-C at any time to re-stiffen and exit. <<<")
    input("Press Enter to connect to the arm...")

    # Stiff bring-up: holds current pose with default PD; nothing floats yet.
    robot = get_yam_robot(channel=args.channel, gripper_type=gripper_type, zero_gravity_mode=False)
    info = robot.get_robot_info()
    n_motors = robot.num_dofs()
    gripper_index = info.get("gripper_index", None)

    factor = parse_factor(args.gcomp, n_motors, gripper_index)
    robot.gravity_comp_factor = factor

    stock_kp = np.asarray(info["kp"], dtype=float).copy()
    stock_kd = np.asarray(info["kd"], dtype=float).copy()

    kdl = MuJoCoKDL(xml_path)

    print(f"\nrobot up. n_motors={n_motors} gripper_index={gripper_index}")
    print(f"  stock kp        = {np.round(stock_kp, 2)}")
    print(f"  stock kd        = {np.round(stock_kd, 2)}")
    print(f"  gravity_comp_factor = {np.round(factor, 3)}")

    def restiffen():
        """Return the arm to a stiff hold at its current pose."""
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

    def log_once():
        obs = robot.get_observations()
        q = np.asarray(obs["joint_pos"], dtype=float)[:ARM_DOF]
        vel = np.asarray(obs["joint_vel"], dtype=float)[:ARM_DOF]
        eff = np.asarray(obs["joint_eff"], dtype=float)[:ARM_DOF]
        g = kdl.compute_inverse_dynamics(q, np.zeros(ARM_DOF), np.zeros(ARM_DOF))
        g_applied = g * factor[:ARM_DOF]
        print("-" * 72)
        print(fmt_row("pos (rad)", q))
        print(fmt_row("vel (rad/s)", vel))
        print(fmt_row("eff measured", eff))
        print(fmt_row("g(q) model", g))
        print(fmt_row("g*factor sent", g_applied))
        print(fmt_row("eff - g*factor", eff - g_applied))  # ~0 when factor is right & arm is still
        return q, vel

    # ---- Phase A: HOLD + measure ----
    print("\n[Phase A] Holding pose. Logging measured effort vs model gravity g(q).")
    print("          Move the arm BY HAND to a few poses; watch 'eff - g*factor'.")
    t_end = time.perf_counter() + 6.0
    while time.perf_counter() < t_end:
        log_once()
        time.sleep(dt)

    if args.measure_only:
        print("\n--measure-only: done. Re-stiffening. SUPPORT THE ARM, then this exits.")
        restiffen()
        time.sleep(0.3)
        return 0

    # ---- Phase B: damped gravity float ----
    print("\n[Phase B] Float test. Arm joints -> kp=0 (kept damped); gripper stays held.")
    print(f"          Runaway guard: |vel|>{args.max_vel} rad/s or drift>{args.max_drift} rad -> abort.")
    ans = input("Type 'float' then Enter to RELEASE the arm into gravity-float (anything else = quit): ").strip()
    if ans.lower() != "float":
        print("Not floating. Re-stiffening and exiting.")
        restiffen()
        time.sleep(0.3)
        return 0

    kp_float = stock_kp.copy()
    kd_float = stock_kd.copy()
    kp_float[:ARM_DOF] = 0.0
    if args.kd_damp is not None:
        kd_float[:ARM_DOF] = args.kd_damp

    start_q = np.asarray(robot.get_joint_pos(), dtype=float)[:ARM_DOF]
    print(f"  released at pos = {np.round(start_q, 3)}")
    print("  >>> WATCH THE ARM. Does it rise (factor too high), sag (too low), or hold (good)? <<<")

    next_log = time.perf_counter()
    try:
        while True:
            full_pos = np.asarray(robot.get_joint_pos(), dtype=float)
            vel_full = np.asarray(robot.get_observations()["joint_vel"], dtype=float)
            cmd = {
                "pos": full_pos,            # ignored for arm (kp=0); harmless
                "vel": np.zeros(n_motors),  # damping target
                "kp": kp_float,
                "kd": kd_float,
            }
            robot.command_joint_state(cmd)

            arm_pos = full_pos[:ARM_DOF]
            arm_vel = vel_full[:ARM_DOF]
            drift = np.abs(arm_pos - start_q)
            if np.any(np.abs(arm_vel) > args.max_vel) or np.any(drift > args.max_drift):
                bad = int(np.argmax(np.maximum(np.abs(arm_vel) / args.max_vel, drift / args.max_drift)))
                print(f"\n!! RUNAWAY GUARD tripped on joint {bad} "
                      f"(vel={arm_vel[bad]:+.3f}, drift={drift[bad]:.3f}) — re-stiffening.")
                restiffen()
                break

            now = time.perf_counter()
            if now >= next_log:
                log_once()
                next_log = now + dt
            time.sleep(0.002)  # ~500 Hz command loop
    except KeyboardInterrupt:
        print("\nCtrl-C in float — re-stiffening.")
        restiffen()

    print("\nDone. Arm is in stiff hold. SUPPORT THE ARM — it will settle when this process exits.")
    input("Press Enter to exit (keep holding the arm)...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
