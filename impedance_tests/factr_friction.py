"""FACTR-style joint friction compensation for the YAM arm (pure numpy, no CAN).

Geared YAM joints (esp. the DM4340 base/shoulder/elbow) have substantial breakaway
(static) friction. In Cartesian impedance leader-follower the impedance torque mapped
onto those joints (~0.5-1.6 N*m) is below that breakaway, so they never move and the
follower stalls. This module computes a per-joint feedforward torque that cancels most
of the friction so modest impedance torques can drive the heavy joints.

Control law (per joint i), the FACTR-style smoothed blend:

    s_v   = tanh(qdot_i / v_eps_i)            # smooth sign of velocity   (-> 0 at rest)
    s_d   = tanh(tau_des_i / t_eps_i)         # smooth sign of desired (impedance) torque
    a_i   = exp(-(qdot_i / v_eps_i)**2)       # at-rest gate: ~1 at qdot~0, -> 0 when moving
    dir_i = (1 - a_i) * s_v + a_i * s_d       # desired-dir at rest (breakaway), vel-dir at speed
    tau_fric_i = enable * (mu_c_i * dir_i + mu_v_i * qdot_i)
    tau_fric_i = clip(tau_fric_i, -fric_max_i, +fric_max_i)

`tau_des` is the PRE-friction, per-joint clipped impedance torque (never feed friction
into its own direction). At rest the term is mu_c*sign(tau_imp) -> breakaway assist in the
commanded direction; once moving it becomes mu_c*sign(qdot)+mu_v*qdot, independent of
tau_imp, so there is no positive feedback at speed (prevents limit-cycle buzz).

STABILITY: keep mu_c strictly below the true breakaway friction tau_s (use ~0.5-0.8x the
calibrated value). Then the comp alone cannot break a joint loose at rest (no self-drive),
but it lowers the effective breakaway threshold to (tau_s - mu_c) so the impedance torque
gets the joint moving. mu_v is NEGATIVE damping (it cancels physical damping) -> default 0;
if used, require mu_v < kd so net joint damping stays positive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARM_DOF = 6

# Per-motor firmware torque limits for the YAM arm joints, from
# i2rt.motor_drivers.utils.MotorType.get_motor_constants and the motor layout in
# i2rt.robots.get_robot.get_yam_robot: joints 0,1,2 = DM4340 (28 N*m),
# joints 3,4,5 = DM4310 (10 N*m). Mirrored here so this module stays import-light
# (no CAN / i2rt dependency, so it works in --selftest with no hardware).
TORQUE_MAX = np.array([28.0, 28.0, 28.0, 10.0, 10.0, 10.0])

# Worst-case gravity torque per joint (|g(q)| upper bound, from measured/model values)
# used only for the budget check below.
GRAVITY_MAX = np.array([0.5, 8.0, 6.0, 1.5, 0.2, 0.2])


@dataclass
class FrictionParams:
    """Per-joint friction-compensation parameters (each a length-6 array)."""

    mu_c: np.ndarray      # Coulomb / breakaway feedforward magnitude (N*m)
    mu_v: np.ndarray      # viscous feedforward gain (N*m / (rad/s)); cancels damping -> keep small
    v_eps: np.ndarray     # velocity scale for the tanh / at-rest gate (rad/s)
    t_eps: np.ndarray     # desired-torque scale for the breakaway tanh (N*m)
    fric_max: np.ndarray  # saturation on |tau_fric| per joint (N*m)

    def __post_init__(self):
        for name in ("mu_c", "mu_v", "v_eps", "t_eps", "fric_max"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (ARM_DOF,):
                raise ValueError(f"FrictionParams.{name} must be length {ARM_DOF}, got {arr.shape}")
            setattr(self, name, arr)


def default_yam_params() -> FrictionParams:
    """Tuned on the YAM follower 2026-06-18 (leader-follower iter6 — smooth, stable, no
    limit cycle). mu_c is RAW; effective = mu_c * mu_scale (test_06 default mu_scale=0.5
    -> eff [0.5, 2.4, 2.6, 0.5, 0.13, 0.11]). v_eps=0.15 keeps the velocity-comp slope
    gentle so the heavy joints break loose when loaded without limit-cycling at light
    poses. Re-run test_05_friction_calibrate.py if the arm or end-effector payload changes.
    """
    return FrictionParams(
        #       base shoulder elbow  wp    wy    wr
        mu_c=  [1.0,  4.8,    5.2,   1.0,  0.25, 0.22],
        mu_v=  [0.0,  0.0,    0.0,   0.0,  0.0,  0.0],
        v_eps= [0.15, 0.15,   0.15,  0.15, 0.15, 0.15],
        t_eps= [0.3,  0.5,    0.5,   0.2,  0.2,  0.2],
        fric_max=[1.5, 3.0,   3.0,   1.0,  1.0,  1.0],
    )


def default_tau_clip() -> np.ndarray:
    """Per-joint clamp on the impedance torque tau_imp (replaces the old flat 4 N*m).
    Sized so tau_clip + fric_max + gravity_max <= 0.85 * TORQUE_MAX (static worst case)."""
    return np.array([8.0, 10.0, 10.0, 4.0, 3.5, 3.5])


def friction_torque(
    qdot: np.ndarray,
    tau_des: np.ndarray,
    params: FrictionParams,
    enable: float = 1.0,
) -> np.ndarray:
    """Per-joint friction-compensation feedforward torque (length 6).

    Args:
        qdot: measured joint velocities (rad/s), length 6 (arm joints only).
        tau_des: PRE-friction impedance torque per joint (N*m), already clipped.
        params: FrictionParams.
        enable: global 0..1 ramp gain (fade comp in after engage / restiffen).
    """
    qdot = np.asarray(qdot, dtype=float)
    tau_des = np.asarray(tau_des, dtype=float)
    p = params
    qn = qdot / p.v_eps
    s_v = np.tanh(qn)
    s_d = np.tanh(tau_des / p.t_eps)
    at_rest = np.exp(-(qn ** 2))                 # ~1 at qdot~0, ->0 when moving
    direction = (1.0 - at_rest) * s_v + at_rest * s_d
    tau_fric = float(enable) * (p.mu_c * direction + p.mu_v * qdot)
    return np.clip(tau_fric, -p.fric_max, p.fric_max)


def check_budget(
    tau_imp: np.ndarray,
    tau_fric: np.ndarray,
    g: np.ndarray,
    kd: np.ndarray,
    qdot: np.ndarray,
    safety: float = 0.9,
) -> list[str]:
    """Return a list of per-joint budget-violation messages (empty = OK).

    Total motor command per joint is kp*(.)+kd*(0-qdot)+(tau_imp+tau_fric+g). With kp=0
    on the arm, require |tau_imp|+|tau_fric|+|g|+|kd*qdot| <= safety*TORQUE_MAX.
    """
    tau_imp = np.asarray(tau_imp, float)
    tau_fric = np.asarray(tau_fric, float)
    g = np.asarray(g, float)
    kd = np.asarray(kd, float)
    qdot = np.asarray(qdot, float)
    total = np.abs(tau_imp) + np.abs(tau_fric) + np.abs(g) + np.abs(kd * qdot)
    limit = safety * TORQUE_MAX
    msgs = []
    for i in range(ARM_DOF):
        if total[i] > limit[i]:
            msgs.append(
                f"joint {i}: |tau|~{total[i]:.1f} > {safety:.2f}*TORQUE_MAX={limit[i]:.1f} N*m"
            )
    return msgs


def friction_curve_demo() -> None:
    """Print tau_fric over a sweep of qdot and tau_des so the blend can be eyeballed
    offline (no hardware). Confirms breakaway-direction at rest, velocity-direction at speed,
    and that |tau_fric| <= fric_max."""
    p = default_yam_params()
    print("FACTR friction curve demo (joint 1 = shoulder, mu_c=%.2f, fric_max=%.2f)"
          % (p.mu_c[1], p.fric_max[1]))
    print(f"{'qdot':>7} {'tau_des':>8} {'tau_fric[1]':>12}   note")
    cases = [
        (0.0, 0.0, "at rest, no command -> ~0 (no self-drive)"),
        (0.0, 0.3, "at rest, small +cmd -> breakaway assist (+)"),
        (0.0, 2.0, "at rest, large +cmd -> ~+mu_c (full breakaway)"),
        (0.0, -2.0, "at rest, large -cmd -> ~-mu_c"),
        (0.5, 0.0, "moving +, no cmd -> +mu_c (cancel kinetic friction)"),
        (0.5, -2.0, "moving + while cmd - -> still +mu_c (velocity wins, no +fb)"),
        (-0.5, 0.0, "moving -, no cmd -> -mu_c"),
        (2.0, 0.0, "fast + -> +mu_c (saturated)"),
    ]
    j = 1
    for qd, td, note in cases:
        qv = np.zeros(ARM_DOF); qv[j] = qd
        tv = np.zeros(ARM_DOF); tv[j] = td
        tf = friction_torque(qv, tv, p)[j]
        print(f"{qd:>7.2f} {td:>8.2f} {tf:>12.3f}   {note}")
    assert np.all(np.abs(friction_torque(np.full(ARM_DOF, 5.0), np.full(ARM_DOF, 5.0), p)) <= p.fric_max + 1e-9)
    print("OK: |tau_fric| <= fric_max for all joints")


if __name__ == "__main__":
    friction_curve_demo()
    print("\ndefault_tau_clip:", default_tau_clip())
    print("TORQUE_MAX      :", TORQUE_MAX)
