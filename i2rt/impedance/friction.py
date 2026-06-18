"""FACTR-style joint friction compensation for the YAM arm (pure numpy, no CAN).

Canonical copy of the validated impedance_tests/factr_friction.py. Geared YAM joints
(esp. the DM4340 base/shoulder/elbow) have substantial breakaway friction; in Cartesian
impedance the J^T F torque mapped onto those joints is below their breakaway, so they
stall. This module computes a per-joint feedforward torque that cancels most of the
friction so modest impedance torques drive the heavy joints.

Control law (per joint i), smoothed velocity/desired-direction blend:

    s_v   = tanh(qdot_i / v_eps_i)            # smooth sign of velocity   (-> 0 at rest)
    s_d   = tanh(tau_des_i / t_eps_i)         # smooth sign of desired (impedance) torque
    a_i   = exp(-(qdot_i / v_eps_i)**2)       # at-rest gate: ~1 at qdot~0, -> 0 when moving
    dir_i = (1 - a_i) * s_v + a_i * s_d       # desired-dir at rest (breakaway), vel-dir at speed
    tau_fric_i = enable * (mu_c_i * dir_i + mu_v_i * qdot_i)
    tau_fric_i = clip(tau_fric_i, -fric_max_i, +fric_max_i)

At rest the term is mu_c*sign(tau_imp) -> breakaway assist; once moving it becomes
mu_c*sign(qdot)+mu_v*qdot, independent of tau_imp -> no positive feedback at speed.
Keep mu_c below the true breakaway friction (stability); v_eps gentle to avoid the
over-comp limit cycle (0.15 validated). Defaults below are the tuned 2026-06-18 (iter6)
leader-follower values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARM_DOF = 6

# Per-motor firmware torque limits for the YAM arm joints (DM4340 0,1,2 = 28 N*m;
# DM4310 3,4,5 = 10 N*m), mirrored so this module needs no i2rt import.
TORQUE_MAX = np.array([28.0, 28.0, 28.0, 10.0, 10.0, 10.0])
GRAVITY_MAX = np.array([0.5, 8.0, 6.0, 1.5, 0.2, 0.2])


@dataclass
class FrictionParams:
    """Per-joint friction-compensation parameters (each length-6)."""

    mu_c: np.ndarray
    mu_v: np.ndarray
    v_eps: np.ndarray
    t_eps: np.ndarray
    fric_max: np.ndarray

    def __post_init__(self):
        for name in ("mu_c", "mu_v", "v_eps", "t_eps", "fric_max"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (ARM_DOF,):
                raise ValueError(f"FrictionParams.{name} must be length {ARM_DOF}, got {arr.shape}")
            setattr(self, name, arr)


def default_yam_params() -> FrictionParams:
    """Tuned on the YAM follower 2026-06-18 (leader-follower iter6). mu_c is RAW;
    effective = mu_c * mu_scale (controller default mu_scale=0.5 -> eff
    [0.5,2.4,2.6,0.5,0.13,0.11]). v_eps=0.15 avoids the over-comp limit cycle."""
    return FrictionParams(
        mu_c=[1.0, 4.8, 5.2, 1.0, 0.25, 0.22],
        mu_v=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        v_eps=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
        t_eps=[0.3, 0.5, 0.5, 0.2, 0.2, 0.2],
        fric_max=[1.5, 3.0, 3.0, 1.0, 1.0, 1.0],
    )


def default_tau_clip() -> np.ndarray:
    """Per-joint clamp on the impedance torque tau_imp (DM4340 vs DM4310 budgets)."""
    return np.array([8.0, 10.0, 10.0, 4.0, 3.5, 3.5])


def friction_torque(qdot, tau_des, params: FrictionParams, enable: float = 1.0) -> np.ndarray:
    """Per-joint friction-comp feedforward torque (length 6). tau_des = PRE-friction
    impedance torque (already clipped). enable = 0..1 fade-in gain."""
    qdot = np.asarray(qdot, dtype=float)
    tau_des = np.asarray(tau_des, dtype=float)
    p = params
    qn = qdot / p.v_eps
    s_v = np.tanh(qn)
    s_d = np.tanh(tau_des / p.t_eps)
    at_rest = np.exp(-(qn ** 2))
    direction = (1.0 - at_rest) * s_v + at_rest * s_d
    tau_fric = float(enable) * (p.mu_c * direction + p.mu_v * qdot)
    return np.clip(tau_fric, -p.fric_max, p.fric_max)
