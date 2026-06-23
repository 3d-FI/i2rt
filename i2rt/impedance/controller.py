"""Server-side leader-follower Cartesian impedance controller for one YAM arm pair.

Runs as a background thread that is the SOLE owner of the follower arm: it reads the
leader and follower joint state locally (no RPC), computes Cartesian impedance + FACTR
friction compensation, and commands the follower via command_joint_torque at ~250 Hz.
The leader floats (its own gravity comp); the follower tracks the leader's EE pose.

This is the validated iter6 controller (impedance_tests/test_06) packaged for use inside
run_bimanual_yam_server.py so the loop is decoupled from cameras/recording.

    x_des = FK(leader_q) + clutch_offset            # clutch captured at engage (no lunge)
    F     = K*(x_des - x_follower) - D*x_dot         # Cartesian wrench
    tau   = clip(J^T F, tau_clip) + friction_comp    # gravity added by i2rt
    command_joint_torque(tau, kp=0 arm, kd=damping, pos=hold/gripper-track)
"""

from __future__ import annotations

import threading
import time

import numpy as np

from i2rt.impedance.friction import default_yam_params, default_tau_clip, friction_torque
from i2rt.impedance.kinematics import MjKin

ARM_DOF = 6


class LeaderFollowerImpedance:
    """Cartesian-impedance leader-follower control for one arm pair (server-side thread)."""

    def __init__(
        self,
        follower,
        leader,
        xml_path: str,
        name: str = "arm",
        *,
        K=(120.0, 120.0, 120.0, 15.0, 15.0, 15.0),
        D=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        arm_kd=(3.0, 4.5, 4.5, 3.0, 2.0, 2.0),
        gcomp_factor: float = 1.0,
        mu_scale: float = 0.5,
        friction: bool = True,
        track_gripper: bool = True,
        max_vel: float = 3.0,
        max_track: float = 0.3,
        workspace: float = 0.25,
        fric_ramp: float = 2.0,
        rate_hz: float = 250.0,
        vel_ff: float = 0.0,
        lookahead: float = 0.0,
        vel_ff_alpha: float = 0.3,
        ki: float = 0.0,
        ei_clip: float = 0.1,
        pose_source=None,
    ):
        self.follower = follower
        self.leader = leader
        self.name = name
        # SpaceMouse (or any) pose source: if set, the desired EE pose comes from
        # pose_source.get_pose() -> (pos(3), R(3x3), grip) instead of a leader arm. No leader
        # joints exist in this mode, so the leader-velocity feedforward is skipped.
        self.pose_source = pose_source
        self.kin = MjKin(xml_path)
        # Leader-velocity feedforward (cuts velocity-proportional following lag):
        #   vel_ff    - gain on leader joint-velocity feedforward torque (0 = off)
        #   lookahead - seconds to advance the Cartesian target by leader EE velocity (0 = off)
        #   vel_ff_alpha - EMA weight smoothing the (noisy) leader velocity
        self.vel_ff = float(vel_ff)
        self.lookahead = float(lookahead)
        self.vel_ff_alpha = float(np.clip(vel_ff_alpha, 0.0, 1.0))
        self._vl_filt = np.zeros(ARM_DOF)
        # Integral term on the Cartesian pose error: removes the steady-state offset a PD
        # spring leaves (gravity-comp residual + friction), so the follower exactly repeats
        # the leader. ei_clip is the anti-windup clamp on the accumulated error.
        self.Ki = float(ki)
        self.ei_clip = float(ei_clip)
        self._ei = np.zeros(6)
        self.K = np.asarray(K, float)
        self.D = np.asarray(D, float)
        self.arm_kd = np.asarray(arm_kd, float)
        self.tau_clip = default_tau_clip()
        self.fp = default_yam_params()
        self.fp.mu_c = self.fp.mu_c * mu_scale
        self.friction = friction
        self.track_gripper = track_gripper
        self.max_vel = max_vel
        self.max_track = max_track
        self.workspace = workspace
        self.fric_ramp = fric_ramp
        self.dt = 1.0 / max(rate_hz, 1.0)
        self.tau_total_clip = self.tau_clip + self.fp.fric_max

        info = follower.get_robot_info()
        self.n = follower.num_dofs()
        self.gripper_index = info.get("gripper_index", None)
        self.stock_kp = np.asarray(info["kp"], float).copy()
        self.stock_kd = np.asarray(info["kd"], float).copy()

        # Validated gravity comp for a follower holding position (1.0, not the buoyant 1.3).
        factor = np.ones(self.n)
        factor[:ARM_DOF] = gcomp_factor
        follower.gravity_comp_factor = factor

        self._stop = threading.Event()
        self._thread = None
        self.restiffen()  # bring the follower up holding its current pose, safely

    # ---- helpers ----
    def _follower_arm(self):
        obs = self.follower.get_observations()
        return (np.asarray(obs["joint_pos"], float)[:ARM_DOF],
                np.asarray(obs["joint_vel"], float)[:ARM_DOF])

    def _leader_arm_grip(self):
        obs = self.leader.get_observations()
        q = np.asarray(obs["joint_pos"], float)[:ARM_DOF]
        qd = np.asarray(obs.get("joint_vel", np.zeros(ARM_DOF)), float)[:ARM_DOF]
        grip = None
        if "gripper_pos" in obs:
            g = np.asarray(obs["gripper_pos"], float).ravel()
            grip = float(g[0]) if g.size else None
        return q, qd, grip

    def _target_pose(self):
        """Desired EE pose + gripper (+ leader joints/velocity) for this tick.

        From the SpaceMouse pose_source if set (no leader arm; ql/qld are None, so the
        leader-velocity feedforward is skipped), else from the leader arm via FK.
        Returns (xl_pos(3), Rl(3x3), grip, ql|None, qld|None).
        """
        if self.pose_source is not None:
            pos, R, grip = self.pose_source.get_pose()
            return np.asarray(pos, float), np.asarray(R, float), grip, None, None
        ql, qld, grip = self._leader_arm_grip()
        xl_pos, Rl = self.kin.fk(ql)
        return xl_pos, Rl, grip, ql, qld

    def restiffen(self):
        try:
            self.follower.update_kp_kd(self.stock_kp, self.stock_kd)
            self.follower.command_joint_pos(self.follower.get_joint_pos())
        except Exception as e:  # noqa: BLE001
            print(f"[{self.name}] restiffen warning: {e}")

    def _clutch(self):
        """Capture the offset so the follower does not lunge to match the target."""
        qf, _ = self._follower_arm()
        xl_pos, Rl, _, _, _ = self._target_pose()
        xf_pos, Rf = self.kin.fk(qf)
        self.pos_offset = xf_pos - xl_pos
        self.R_offset = Rl.T @ Rf
        self.box_lo = xf_pos - self.workspace
        self.box_hi = xf_pos + self.workspace
        self.t_engage = time.perf_counter()
        self._ei = np.zeros(6)  # reset integral so a re-clutch never carries stale accumulation

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"impedance_{self.name}")
        self._thread.start()
        return self._thread

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.restiffen()

    # ---- control loop ----
    def _run(self):
        kp_full = np.zeros(self.n)
        kd_full = np.zeros(self.n); kd_full[:ARM_DOF] = self.arm_kd
        if self.gripper_index is not None:
            kp_full[self.gripper_index] = self.stock_kp[self.gripper_index]
            kd_full[self.gripper_index] = self.stock_kd[self.gripper_index]

        self._clutch()
        src = "SpaceMouse" if self.pose_source is not None else "leader"
        print(f"[{self.name}] impedance engaged (clutched). Move the {src}; follower tracks.")
        while not self._stop.is_set():
            _iter_t = time.perf_counter()
            try:
                xl_pos, Rl, grip, ql, qld = self._target_pose()
                if qld is not None:
                    # Leader mode: EMA-filter the noisy leader velocity for feedforward, and
                    # advance the goal by leader EE velocity * lookahead (cancels velocity-prop lag).
                    self._vl_filt = (1.0 - self.vel_ff_alpha) * self._vl_filt + self.vel_ff_alpha * qld
                    qld_f = self._vl_filt
                    if self.lookahead > 0.0:
                        xl_pos = xl_pos + (self.kin.jacobian(ql) @ qld_f)[:3] * self.lookahead
                else:
                    qld_f = None  # pose_source (SpaceMouse): no leader velocity -> no feedforward
                x_des_pos = np.clip(xl_pos + self.pos_offset, self.box_lo, self.box_hi)
                R_des = Rl @ self.R_offset

                qf, qdf = self._follower_arm()
                full_pos = np.asarray(self.follower.get_joint_pos(), float)
                J = self.kin.jacobian(qf)
                xdot = J @ qdf
                dx = self.kin.pose_error(x_des_pos, R_des, qf)
                if self.Ki > 0.0:
                    self._ei = np.clip(self._ei + dx * self.dt, -self.ei_clip, self.ei_clip)
                    F = self.K * dx + self.Ki * self._ei - self.D * xdot
                else:
                    F = self.K * dx - self.D * xdot
                tau_imp = np.clip(J.T @ F, -self.tau_clip, self.tau_clip)
                if self.friction:
                    enable = min(1.0, (time.perf_counter() - self.t_engage) / max(self.fric_ramp, 1e-3))
                    tau_fric = friction_torque(qdf, tau_imp, self.fp, enable)
                else:
                    tau_fric = np.zeros(ARM_DOF)
                # Leader-velocity feedforward: net follower velocity term becomes
                # arm_kd*(vel_ff*q̇_leader - q̇_follower) instead of arm_kd*(-q̇_follower), i.e. the
                # follower actively matches the leader's joint velocity rather than only self-damping.
                tau_velff = self.vel_ff * self.arm_kd * qld_f if (qld_f is not None and self.vel_ff > 0.0) else 0.0
                tau_arm = np.clip(tau_imp + tau_fric + tau_velff, -self.tau_total_clip, self.tau_total_clip)

                track = np.linalg.norm(dx[:3])
                if track > self.max_track or np.any(np.abs(qdf) > self.max_vel):
                    print(f"[{self.name}] guard (track={track*100:.0f}cm vel={np.abs(qdf).max():.1f}) "
                          f"-> re-stiffen + re-clutch")
                    self.restiffen()
                    time.sleep(0.3)
                    self._clutch()
                    continue

                torque_full = np.zeros(self.n)
                torque_full[:ARM_DOF] = tau_arm
                pos_full = full_pos.copy()
                if self.track_gripper and self.gripper_index is not None and grip is not None:
                    pos_full[self.gripper_index] = grip  # follower gripper tracks leader handle
                self.follower.command_joint_torque(torque_full, kp=kp_full, kd=kd_full, pos=pos_full)
            except Exception as e:  # noqa: BLE001
                print(f"[{self.name}] loop error: {e} -> re-stiffen")
                self.restiffen()
                time.sleep(0.2)
                self._clutch()
            # Rate-compensated pacing: sleep only the remainder of the period so the loop
            # holds rate_hz (250Hz) instead of always adding a full dt on top of the work.
            time.sleep(max(0.0, self.dt - (time.perf_counter() - _iter_t)))
        print(f"[{self.name}] impedance stopped.")
