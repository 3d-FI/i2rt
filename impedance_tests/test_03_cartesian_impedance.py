#!/usr/bin/env python3
"""Test 03 — Cartesian impedance on a single YAM arm.

Control law (gravity handled by i2rt, so we inject only the impedance wrench):

    x      = forward kinematics of the 'grasp_site'  (MuJoCo, same yam model)
    J      = 6x6 site Jacobian (mj_jacSite), world-aligned
    x_dot  = J @ qdot
    dx     = [ x_des_pos - x_pos ,  log( R_des @ R^T ) ]      # 6D pose error
    F      = K * dx  -  D * x_dot                              # Cartesian wrench
    tau    = J^T @ F                                           # joint torque
    -> command_joint_torque(tau, kd=joint_damping)            # i2rt adds g(q)

Start with x_des = the pose at engage time, so the arm holds its Cartesian pose
and behaves like a 3D spring: push the gripper in any direction and it pushes
back along that direction and returns.

SAFETY
  - Brings up STIFF; you opt into impedance by typing 'engage' (x_des = current
    pose, so the wrench starts at ~0 -> no jump).
  - Per-joint clamp on the impedance torque (|J^T F| <= --tau-clip); the motors
    additionally hard-clip total torque, and gravity is added on top.
  - Joint-space damping (kd) gives a stable damping floor independent of J.
  - Runaway guard re-stiffens on excess EE drift, rotation, or joint velocity.
  - Gripper stays position-held. Push GENTLY first; keep a hand ready; Ctrl-C
    re-stiffens. SUPPORT THE ARM before the process exits.

OFFLINE CHECK (no hardware, no CAN):
    python test_03_cartesian_impedance.py --selftest
  Loads the model and verifies the Jacobian against finite differences of FK.

HARDWARE:
    python test_03_cartesian_impedance.py --channel can_follower --gripper linear_4310
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
    """Forward kinematics + 6x6 site Jacobian for the YAM arm from its MuJoCo model."""

    def __init__(self, xml_path: str, site: str = "grasp_site"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError(f"site '{site}' not found in {xml_path}")
        self.nv = self.model.nv

    def _set(self, q: np.ndarray) -> None:
        self.data.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # required before mj_jacSite

    def fk(self, q: np.ndarray):
        self._set(q)
        pos = self.data.site_xpos[self.site_id].copy()
        R = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, R

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        self._set(q)
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return np.vstack([jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]])  # 6x6


def pose_error(x_des_pos, R_des, x_pos, R):
    pos_err = x_des_pos - x_pos
    rot_err = Rotation.from_matrix(R_des @ R.T).as_rotvec()  # world-frame axis*angle
    return np.concatenate([pos_err, rot_err])


def selftest(xml_path: str) -> int:
    """Finite-difference check of the Jacobian against FK. No hardware."""
    print(f"[selftest] loading {xml_path}")
    kin = MjKin(xml_path)
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(5):
        q = rng.uniform(-0.5, 0.5, ARM_DOF)
        J = kin.jacobian(q)
        p0, R0 = kin.fk(q)
        eps = 1e-6
        Jfd = np.zeros((6, ARM_DOF))
        for i in range(ARM_DOF):
            dq = np.zeros(ARM_DOF); dq[i] = eps
            p1, R1 = kin.fk(q + dq)
            Jfd[:3, i] = (p1 - p0) / eps
            Jfd[3:, i] = Rotation.from_matrix(R1 @ R0.T).as_rotvec() / eps
        err = np.abs(J - Jfd).max()
        worst = max(worst, err)
        print(f"  trial {trial}: max|J - J_fd| = {err:.2e}")
    ok = worst < 1e-3
    print(f"[selftest] worst error {worst:.2e} -> {'PASS' if ok else 'FAIL'}")
    print(f"  FK at q=0: pos={np.round(kin.fk(np.zeros(ARM_DOF))[0], 4)}")
    return 0 if ok else 1


def parse_vec(s: str, n: int, name: str) -> np.ndarray:
    parts = [float(x) for x in s.split(",") if x.strip() != ""]
    if len(parts) == 1:
        return np.full(n, parts[0])
    if len(parts) == n:
        return np.array(parts)
    raise ValueError(f"--{name} expects 1 or {n} values, got {len(parts)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="YAM Cartesian impedance test")
    ap.add_argument("--selftest", action="store_true", help="offline FK/Jacobian check, no hardware")
    ap.add_argument("--channel", default="can_follower")
    ap.add_argument("--gripper", default="linear_4310",
                    choices=["crank_4310", "linear_3507", "linear_4310", "no_gripper"])
    ap.add_argument("--gcomp", default="1.0", help="gravity_comp_factor (validated = 1.0)")
    ap.add_argument("--k-trans", default="120,120,120", help="Cartesian stiffness N/m (x,y,z)")
    ap.add_argument("--k-rot", default="6,6,6", help="Cartesian stiffness N*m/rad (rx,ry,rz)")
    ap.add_argument("--d-trans", default="0,0,0", help="Cartesian damping N*s/m (default 0; use joint kd)")
    ap.add_argument("--d-rot", default="0,0,0", help="Cartesian rotational damping")
    ap.add_argument("--kd", default="3,3,3,2,1,1", help="joint-space damping floor (<=5)")
    ap.add_argument("--tau-clip", type=float, default=4.0, help="clamp on |J^T F| per joint (N*m)")
    ap.add_argument("--max-pos-drift", type=float, default=0.15, help="guard: EE position drift (m)")
    ap.add_argument("--max-rot-drift", type=float, default=0.6, help="guard: EE rotation drift (rad)")
    ap.add_argument("--max-vel", type=float, default=2.0, help="guard: max |joint vel| rad/s")
    ap.add_argument("--rate", type=float, default=4.0, help="console log rate (Hz)")
    args = ap.parse_args()

    gripper_type = GripperType(args.gripper)
    xml_path = gripper_type.get_xml_path()

    if args.selftest:
        return selftest(xml_path)

    K = np.concatenate([parse_vec(args.k_trans, 3, "k-trans"), parse_vec(args.k_rot, 3, "k-rot")])
    D = np.concatenate([parse_vec(args.d_trans, 3, "d-trans"), parse_vec(args.d_rot, 3, "d-rot")])
    arm_kd = parse_vec(args.kd, ARM_DOF, "kd")

    kin = MjKin(xml_path)

    print("=" * 72)
    print("YAM Cartesian impedance test")
    print(f"  channel : {args.channel}   gripper: {args.gripper}")
    print(f"  K       : {K}")
    print(f"  D (cart): {D}")
    print(f"  kd joint: {arm_kd}   tau_clip: {args.tau_clip} N*m")
    print("  bringup : STIFF HOLD — arm will NOT move on start")
    print("=" * 72)
    print(">>> Keep a hand near the arm. Push GENTLY first. Ctrl-C re-stiffens. <<<")
    input("Press Enter to connect...")

    # Heavy import deferred so --selftest needs no CAN stack.
    from i2rt.robots.get_robot import get_yam_robot

    robot = get_yam_robot(channel=args.channel, gripper_type=gripper_type, zero_gravity_mode=False)
    info = robot.get_robot_info()
    n_motors = robot.num_dofs()
    gripper_index = info.get("gripper_index", None)
    stock_kp = np.asarray(info["kp"], dtype=float).copy()
    stock_kd = np.asarray(info["kd"], dtype=float).copy()

    factor = np.ones(n_motors)
    gparts = [float(x) for x in args.gcomp.split(",") if x.strip() != ""]
    factor[:ARM_DOF] = gparts[0] if len(gparts) == 1 else np.array(gparts)[:ARM_DOF]
    robot.gravity_comp_factor = factor

    def restiffen():
        try:
            robot.update_kp_kd(stock_kp, stock_kd)
            robot.command_joint_pos(robot.get_joint_pos())
        except Exception as e:  # noqa: BLE001
            print(f"  (restiffen warning: {e})")

    def on_sigint(signum, frame):  # noqa: ARG001
        print("\nCtrl-C — re-stiffening. SUPPORT THE ARM.")
        restiffen(); time.sleep(0.3); sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    # FK sanity cross-check against i2rt's mink-based kinematics on the same model.
    q0 = np.asarray(robot.get_observations()["joint_pos"], dtype=float)[:ARM_DOF]
    my_pos, _ = kin.fk(q0)
    try:
        from i2rt.robots.kinematics import Kinematics
        ref_T = Kinematics(xml_path, "grasp_site").fk(q0)
        diff = np.linalg.norm(my_pos - ref_T[:3, 3])
        print(f"FK cross-check vs i2rt: my={np.round(my_pos,4)} ref={np.round(ref_T[:3,3],4)} diff={diff*1000:.2f} mm")
        if diff > 5e-3:
            print("!! FK mismatch > 5 mm — model/site wrong. Re-stiffening and aborting.")
            restiffen(); time.sleep(0.3); return 1
    except Exception as e:  # noqa: BLE001
        print(f"(FK cross-check skipped: {e})")

    print("\n[Phase A] Stiff hold.")
    t_end = time.perf_counter() + 4.0
    while time.perf_counter() < t_end:
        time.sleep(0.05)

    ans = input("Type 'engage' + Enter to start Cartesian impedance at the current pose: ").strip()
    if ans.lower() != "engage":
        print("Not engaging. Re-stiffening.")
        restiffen(); time.sleep(0.3); return 0

    q_now = np.asarray(robot.get_observations()["joint_pos"], dtype=float)[:ARM_DOF]
    x_des_pos, R_des = kin.fk(q_now)
    kp_full = np.zeros(n_motors)
    kd_full = np.zeros(n_motors); kd_full[:ARM_DOF] = arm_kd
    if gripper_index is not None:
        kp_full[gripper_index] = stock_kp[gripper_index]
        kd_full[gripper_index] = stock_kd[gripper_index]

    print(f"\n[Phase B] CARTESIAN IMPEDANCE. Target EE pos = {np.round(x_des_pos, 4)}")
    print("  >>> Gently push the GRIPPER in x/y/z — it should push back & return. <<<")
    dt = 1.0 / max(args.rate, 0.5)
    next_log = time.perf_counter()
    try:
        while True:
            obs = robot.get_observations()
            q = np.asarray(obs["joint_pos"], dtype=float)[:ARM_DOF]
            qdot = np.asarray(obs["joint_vel"], dtype=float)[:ARM_DOF]
            full_pos = np.asarray(robot.get_joint_pos(), dtype=float)

            x_pos, R = kin.fk(q)
            J = kin.jacobian(q)
            x_dot = J @ qdot
            dx = pose_error(x_des_pos, R_des, x_pos, R)
            F = K * dx - D * x_dot
            tau = J.T @ F
            tau = np.clip(tau, -args.tau_clip, args.tau_clip)

            pos_drift = np.linalg.norm(dx[:3])
            rot_drift = np.linalg.norm(dx[3:])
            if pos_drift > args.max_pos_drift or rot_drift > args.max_rot_drift \
                    or np.any(np.abs(qdot) > args.max_vel):
                print(f"\n!! RUNAWAY GUARD: pos_drift={pos_drift*100:.1f}cm "
                      f"rot_drift={rot_drift:.2f}rad max_vel={np.abs(qdot).max():.2f} — re-stiffening.")
                restiffen(); break

            torque_full = np.zeros(n_motors)
            torque_full[:ARM_DOF] = tau
            robot.command_joint_torque(torque_full, kp=kp_full, kd=kd_full, pos=full_pos)

            now = time.perf_counter()
            if now >= next_log:
                print("-" * 72)
                print("EE pos    ", np.round(x_pos, 4), " drift(cm)", round(pos_drift * 100, 1))
                print("pos_err(m)", np.round(dx[:3], 4), " rot_err  ", np.round(dx[3:], 3))
                print("wrench F  ", np.round(F, 2))
                print("tau J^T F ", np.round(tau, 2))
                next_log = now + dt
            time.sleep(0.005)  # ~200 Hz
    except KeyboardInterrupt:
        print("\nCtrl-C — re-stiffening.")
        restiffen()

    print("\nDone. Arm in stiff hold. SUPPORT THE ARM — it relaxes when the process exits.")
    input("Press Enter to exit (keep holding the arm)...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
