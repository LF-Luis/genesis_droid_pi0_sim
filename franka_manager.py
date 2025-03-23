import math
from typing import Tuple

import torch
import numpy as np
import genesis as gs

from sim_utils.transformations import quaternion_multiply


MUJOCO_FILE = "xml/franka_emika_panda/panda.xml"
BASE_POS = (0, 0, 0)
# Robot joints: 7 arm joints + 2 finger joints
JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2"
]
END_EFFECTOR_NAME = "hand"
# Links PD gains
PROPORTIONAL_GAINS = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
VELOCITY_GAINS = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
# Links force ranges
FORCE_RANGES_LOWER = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100])
FORCE_RANGES_UPPER = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])

HOME_POS = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
HOME_POS_STEPS = 50  # Steps to wait for stabilization

# Cam config values
CAM_RES = (1280, 720)  # Zed mini at 60 fps (this is for the wrist cam)
CAM_FOV = 57.  # Vertical FOV for Zed mini: 57 degrees


class FrankaManager:
    """
    Manage a Franka robot and all peripherals (e.g. cameras) attached to it in a given scene.
    """

    def __init__(self, scene):
        """
        Add robot to scene and set control config values.
        """
        self._scene = scene
        self._franka = self._scene.add_entity(gs.morphs.MJCF(file=MUJOCO_FILE, pos=BASE_POS))
        # Wrist camera will be attached to the robot's wrist
        self._wrist_camera = scene.add_camera(
            res=CAM_RES,
            fov=CAM_FOV,
            GUI=True,
        )
        self._end_effector = self._franka.get_link(END_EFFECTOR_NAME)
        self.dofs_idx = [self._franka.get_joint(name).dof_idx_local for name in JOINT_NAMES]
        print(f"Franka's DoF idx: {self.dofs_idx}")
        print(f"Franka Wrist Cam Intrinsics: \n{self._wrist_camera.intrinsics}")

    def _set_control_params(self):
        # Setting control parameters
        self._franka.set_dofs_kp(PROPORTIONAL_GAINS, self.dofs_idx)
        self._franka.set_dofs_kv(VELOCITY_GAINS, self.dofs_idx)
        self._franka.set_dofs_force_range(FORCE_RANGES_LOWER, FORCE_RANGES_UPPER, self.dofs_idx)

    def set_to_init_pos(self):
        """
        Set control params and reset to an initial position.
        """
        self._set_control_params()
        # Teleport to position
        self._franka.set_dofs_position(HOME_POS, self.dofs_idx)
        # Solve for position
        self._franka.control_dofs_position(HOME_POS, self.dofs_idx)

        # Wait for stabilization
        print(f"Running {HOME_POS_STEPS} steps to stabilize at home position.")
        for _ in range(HOME_POS_STEPS):
            self._scene.step()
        print(f"Done waiting for stabilization.")

    def get_ee_pos(self) -> torch.Tensor:
        return self._end_effector.get_pos()

    def get_ee_quat(self) -> torch.Tensor:
        return self._end_effector.get_quat()

    def get_joints_and_gripper_pos(self):
        # Get the current joint and gripper revolute angles in radians
        joint_positions = self._franka.get_dofs_position(self.dofs_idx[:7])  # First 7 DOFs are the arm joints
        gripper_position = self._franka.get_dofs_position(self.dofs_idx[7:8])  # 8th DOF is the first finger joint
        return joint_positions, gripper_position

    def set_revolute_radians(self, action):
        # Set the current joint and gripper revolute angles in radians
        self._franka.control_dofs_position(action, self.dofs_idx)

    def cam_render(self):
        return self._wrist_camera.render()

    def _get_wrist_cam_offset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return translation and rotation offset of wrist camera w.r.t. end-effector.

        Numbers were manually tuned.
        """
        wrist_pos = self.get_ee_pos()

        # Translation
        translation = torch.tensor([0.11, 0.0, -0.04], device=wrist_pos.device)

        # Rotation
        # 180° rotation about x-axis: flip camera's z-axis.
        q_x = torch.tensor([0.0, 1.0, 0.0, 0.0])  # [cos(pi/2), sin(pi/2), 0, 0]
        # 90° rotation about z-axis.
        angle_z = math.pi / 2
        q_z = torch.tensor([math.cos(angle_z/2), 0.0, 0.0, math.sin(angle_z/2)])
        # -20° rotation about y-axis.
        angle_y = math.radians(-20)
        q_y = torch.tensor([math.cos(angle_y/2), 0.0, math.sin(angle_y/2), 0.0])
        # Combine rotations: apply q_x, then q_z, then q_y
        offset_quaternion = quaternion_multiply(q_z, q_x)
        offset_quaternion = quaternion_multiply(q_y, offset_quaternion)


# TODO:
# [ ] move all wrist cam logic (setup and update) into franka_manager.py
