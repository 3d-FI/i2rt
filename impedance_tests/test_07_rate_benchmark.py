#!/usr/bin/env python3
"""Test 07 — Control-loop rate benchmark for the YAM Cartesian impedance loop.

Mirrors the Trossen "Test 1: Command Rate Benchmark" but for the i2rt/YAM architecture,
where there are TWO rates:

  1. i2rt motor thread (MotorChainRobot.update @ ~250 Hz): owns CAN I/O, applies
     gravity + commanded torque to the motors. Fixed, independent of our loop.
  2. our impedance loop: reads CACHED state (get_observations, no CAN round-trip),
     computes FK + 6x6 Jacobian + J^T F + friction comp, writes self._commands.torques
     (command_joint_torque, no CAN). Pure compute -> this benchmark measures its ceiling.

The motor command-application rate is the i2rt thread's ~250 Hz; our loop only needs to
keep up with that (and it does, by a wide margin). The point of running the impedance loop
INSIDE the i2rt server process is that this compute stays isolated from cameras/recording.

OFFLINE (no hardware -- measures the impedance MATH ceiling):
    python test_07_rate_benchmark.py
HARDWARE (full loop incl. cached state read + command):
    python test_07_rate_benchmark.py --channel can_follower --secs 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from factr_friction import default_yam_params, default_tau_clip, friction_torque  # noqa: E402
from i2rt.robots.utils import GripperType  # noqa: E402

ARM_DOF = 6


class MjKin:
    def __init__(self, xml_path, site="grasp_site"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
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
        jp = np.zeros((3, self.nv)); jr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site_id)
        return np.vstack([jp[:, :ARM_DOF], jr[:, :ARM_DOF]])


def impedance_step(kin, fp, K, D, tau_clip, q, qdot, x_des_pos, R_des):
    """One full impedance computation (what runs every loop iteration)."""
    x_pos, R = kin.fk(q)
    J = kin.jacobian(q)
    xdot = J @ qdot
    dx = np.concatenate([x_des_pos - x_pos, Rotation.from_matrix(R_des @ R.T).as_rotvec()])
    F = K * dx - D * xdot
    tau_imp = np.clip(J.T @ F, -tau_clip, tau_clip)
    tau_fric = friction_torque(qdot, tau_imp, fp)
    return np.clip(tau_imp + tau_fric, -(tau_clip + fp.fric_max), tau_clip + fp.fric_max)


def report(name, dts):
    dts = np.array(dts)
    hz = 1.0 / dts.mean()
    print(f"  {name:<32} {hz:>9.0f} Hz   dt mean {dts.mean()*1e3:.3f} ms  p50 {np.percentile(dts,50)*1e3:.3f}  p99 {np.percentile(dts,99)*1e3:.3f}")
    return hz


def main() -> int:
    ap = argparse.ArgumentParser(description="YAM impedance loop-rate benchmark")
    ap.add_argument("--channel", default=None, help="follower CAN (omit = offline math-only benchmark)")
    ap.add_argument("--gripper", default="linear_4310")
    ap.add_argument("--secs", type=float, default=5.0)
    args = ap.parse_args()

    gt = GripperType(args.gripper)
    xml = gt.get_xml_path()
    kin = MjKin(xml)
    fp = default_yam_params()
    K = np.array([120, 120, 120, 15, 15, 15.0])
    D = np.zeros(6)
    tau_clip = default_tau_clip()
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.5, 0.5, ARM_DOF); qdot = rng.uniform(-0.1, 0.1, ARM_DOF)
    x_des, R_des = kin.fk(q)

    print("=" * 72)
    print("YAM impedance loop-rate benchmark")
    print("=" * 72)

    # --- Offline: pure impedance math ceiling (no CAN) ---
    print("\n[A] Impedance MATH only (FK + 6x6 Jacobian + J^T F + friction comp), no hardware:")
    # warmup
    for _ in range(200):
        impedance_step(kin, fp, K, D, tau_clip, q, qdot, x_des, R_des)
    dts = []
    t_end = time.perf_counter() + 2.0
    while time.perf_counter() < t_end:
        t = time.perf_counter()
        impedance_step(kin, fp, K, D, tau_clip, q, qdot, x_des, R_des)
        dts.append(time.perf_counter() - t)
    report("impedance math/iter", dts)
    print("  -> this is the per-arm compute ceiling; bimanual ~= half (two arms in one loop).")

    if args.channel is None:
        print("\n(no --channel: offline benchmark only. Add --channel can_follower for the full hardware loop.)")
        return 0

    # --- Hardware: full loop incl. cached-state read + command write ---
    from i2rt.robots.get_robot import get_yam_robot
    print(f"\n[B] FULL loop on hardware ({args.channel}): read cached state + math + command_joint_torque")
    print(">>> arm comes up STIFF and is NOT commanded to move (x_des = current). Keep clear. <<<")
    input("Press Enter to connect and benchmark...")
    robot = get_yam_robot(channel=args.channel, gripper_type=gt, zero_gravity_mode=False)
    n = robot.num_dofs()
    q0 = np.asarray(robot.get_observations()["joint_pos"], float)[:ARM_DOF]
    x_des, R_des = kin.fk(q0)
    kp = np.zeros(n); kd = np.zeros(n); kd[:ARM_DOF] = [3, 4.5, 4.5, 3, 2, 2]
    pos_hold = np.asarray(robot.get_joint_pos(), float)

    dts = []
    state_changes = 0
    last_pos = None
    t_end = time.perf_counter() + args.secs
    while time.perf_counter() < t_end:
        t = time.perf_counter()
        obs = robot.get_observations()
        q = np.asarray(obs["joint_pos"], float)[:ARM_DOF]
        qd = np.asarray(obs["joint_vel"], float)[:ARM_DOF]
        tau = impedance_step(kin, fp, K, D, tau_clip, q, qd, x_des, R_des)
        torque_full = np.zeros(n); torque_full[:ARM_DOF] = tau
        robot.command_joint_torque(torque_full, kp=kp, kd=kd, pos=pos_hold)
        if last_pos is None or not np.array_equal(q, last_pos):
            state_changes += 1
            last_pos = q
        dts.append(time.perf_counter() - t)

    full_hz = report("full impedance loop", dts)
    fresh_hz = state_changes / args.secs
    print(f"  fresh-state updates (i2rt motor-thread CAN rate): ~{fresh_hz:.0f} Hz")
    print("\nINTERPRETATION:")
    print(f"  - our loop runs at {full_hz:.0f} Hz (compute-bound; cached reads/writes, no CAN I/O).")
    print(f"  - the motors get new torque at the i2rt thread rate (~{fresh_hz:.0f} Hz CAN).")
    print("  - both are >> the 30 Hz camera/recording rate -> put this loop in the i2rt SERVER")
    print("    process so cameras/recording (separate process) never throttle it.")
    print("\nSUPPORT THE ARM. Press Enter to exit.")
    try:
        robot.update_kp_kd(np.asarray(robot.get_robot_info()["kp"], float),
                           np.asarray(robot.get_robot_info()["kd"], float))
        robot.command_joint_pos(robot.get_joint_pos())
        input("")
    except (KeyboardInterrupt, Exception):  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
