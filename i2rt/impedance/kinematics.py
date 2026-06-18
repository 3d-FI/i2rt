"""Forward kinematics + 6x6 site Jacobian for the YAM arm via MuJoCo.

Reuses the same yam.xml MuJoCo model i2rt already ships (so no new dependency: mujoco is
an i2rt dependency). mj_jacSite gives the world-aligned EE Jacobian for J^T F impedance.
"""

from __future__ import annotations

import numpy as np
import mujoco

ARM_DOF = 6


def log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) log map: rotation matrix -> rotation vector (axis * angle). Pure numpy
    (no scipy needed)."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis * (angle / (2.0 * np.sin(angle)))


class MjKin:
    """FK + 6x6 site Jacobian for the YAM arm."""

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
        return (self.data.site_xpos[self.site_id].copy(),
                self.data.site_xmat[self.site_id].reshape(3, 3).copy())

    def jacobian(self, q):
        self._set(q)
        jp = np.zeros((3, self.nv)); jr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site_id)
        return np.vstack([jp[:, :ARM_DOF], jr[:, :ARM_DOF]])

    def pose_error(self, x_des_pos, R_des, q):
        """6-vector EE pose error [pos_err(3), orient_err(3)] at config q."""
        x_pos, R = self.fk(q)
        return np.concatenate([x_des_pos - x_pos, log_so3(R_des @ R.T)])
