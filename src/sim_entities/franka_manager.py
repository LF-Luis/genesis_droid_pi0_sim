import torch
import genesis as gs

from sim_utils.transformations import get_camera_transform
from src.sim_entities.franka_manager_const import (
    MUJOCO_FILE, BASE_POS, JOINT_NAMES, END_EFFECTOR_NAME, PROPORTIONAL_GAINS,
    VELOCITY_GAINS, FORCE_RANGES_LOWER, FORCE_RANGES_UPPER, HOME_POS, HOME_POS_STEPS,
    CAM_RES, CAM_FOV, get_wrist_cam_offset,
)


class FrankaManager:
    """
    Manage a Franka robot and all peripherals (e.g. cameras) attached to it in a given scene.
    """

    def __init__(self, scene):
        """
        Add robot to scene and set control config values.
        """
        self._scene = scene

        # self._franka = self._scene.add_entity(gs.morphs.MJCF(file=MUJOCO_FILE, pos=BASE_POS))

        self._franka = self._scene.add_entity(
            gs.morphs.MJCF(file=MUJOCO_FILE, pos=BASE_POS),
            material=gs.materials.Rigid(
                friction=0.3,  # Lower friction for robot
                coup_restitution=0.0,  # No bouncing during coupling
            ),
            surface=gs.surfaces.Default(vis_mode="visual"),
            # surface=gs.surfaces.Default(vis_mode="collision"),
        )

        self._end_effector = self._franka.get_link(END_EFFECTOR_NAME)
        self.dofs_idx = [self._franka.get_joint(name).dof_idx_local for name in JOINT_NAMES]

        # Wrist camera will be attached to the robot's wrist
        self._wrist_camera = scene.add_camera(
            pos=(0.0, 0.0, 0.0),  # initial pose (will be updated dynamically)
            lookat=(0.0, 0.0, 0.0),
            res=CAM_RES,
            fov=CAM_FOV,
            GUI=True,
        )
        self._cam_ee_pos_offset, self._cam_ee_rot_offset = get_wrist_cam_offset(gs.device)

        print(f"Franka's DoF idx: {self.dofs_idx}")
        print(f"Franka Wrist Cam Intrinsics: \n{self._wrist_camera.intrinsics}")
        print(f"Franka running in device: {gs.device}")

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
        self.step()  # Perform setup once
        print(f"Done waiting for stabilization.")

    def get_ee_pos(self) -> torch.Tensor:
        return self._end_effector.get_pos()

    def get_ee_quat(self) -> torch.Tensor:
        return self._end_effector.get_quat()

    def get_joints_and_gripper_pos(self):
        # Get the current joint and gripper revolute angles in radians
        dofs_positions = self._franka.get_dofs_position()  # 9 joints, held in CUDA Tensor
        joint_positions = dofs_positions[:7]  # First 7 DOFs are the arm joints
        gripper_position = dofs_positions[7:]  # 8th and 9th DOF is the gripper joints
        return joint_positions, gripper_position

    def set_joints_and_gripper_pos(self, action):
        # Set the current joint and gripper positions
        self._franka.control_dofs_position(action, self.dofs_idx)

    def cam_render(self):
        return self._wrist_camera.render()

    def step(self, vis = True):
        """
        Run on each sim step (not tied to fps).
        Set vis = False to skip computing cam location and graphics during movement.
        """
        if vis:
            cam_t = get_camera_transform(
                self.get_ee_pos(),
                self.get_ee_quat(),
                self._cam_ee_pos_offset,
                self._cam_ee_rot_offset
            )
            self._wrist_camera.set_pose(transform=cam_t.cpu().numpy())
            _ = self._wrist_camera.render()
