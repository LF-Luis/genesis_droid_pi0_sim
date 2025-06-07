import numpy as np
from src.sim_entities.franka_manager import FrankaManager

class RobotPoseDebug:
    """
    Other helpful functions:
    _franka.get_qpos()
    """

    def __init__(self, franka_manager: FrankaManager, scene, verbose=False):
        """
        Initialize robot debug layout with movement and rotation controls.

        Args:
            franka_manager: FrankaManager object to control the robot
            scene: Genesis simulation scene object
            verbose: Enable verbose logging
        """
        self._franka_manager = franka_manager
        self._scene = scene
        self._verbose = verbose

        # Define movement vectors in local coordinates
        self._directions = {
            'x-': np.array([-0.1, 0, 0]),
            'x+': np.array([0.1, 0, 0]),
            'y+': np.array([0, 0.1, 0]),
            'y-': np.array([0, -0.1, 0]),
            'z+': np.array([0, 0, 0.1]),
            'z-': np.array([0, 0, -0.1]),
        }

    # Movement shortcuts
    def move_x_plus(self, scale=1.0, grip=None): self._move_end_effector('x+', scale, grip)
    def move_x_minus(self, scale=1.0, grip=None): self._move_end_effector('x-', scale, grip)
    def move_y_plus(self, scale=1.0, grip=None): self._move_end_effector('y+', scale, grip)
    def move_y_minus(self, scale=1.0, grip=None): self._move_end_effector('y-', scale, grip)
    def move_z_plus(self, scale=1.0, grip=None): self._move_end_effector('z+', scale, grip)
    def move_z_minus(self, scale=1.0, grip=None): self._move_end_effector('z-', scale, grip)

    # Rotation shortcuts
    def rotate_x(self, angle=10): self._rotate_end_effector('x', angle)
    def rotate_y(self, angle=10): self._rotate_end_effector('y', angle)
    def rotate_z(self, angle=10): self._rotate_end_effector('z', angle)

    def _move_end_effector(self, direction, scale=1.0, grip=None):
        current_pos = self._franka_manager.get_ee_pos().cpu().numpy()
        move_vector = self._directions[direction] * scale
        new_pos = current_pos + move_vector
        current_quat = self._franka_manager.get_ee_quat().cpu().numpy()

        # Solve IK for the new position
        qpos, ik_diff_err = self._franka_manager._franka.inverse_kinematics(
            link=self._franka_manager._end_effector,
            pos=new_pos,
            quat=current_quat,
            return_error=True
        )

        if self._verbose:
            print(f"qpos: {qpos} \nIK error: {ik_diff_err}")

        # Handle gripper control if specified
        if grip is not None:
            qpos[7:9] = grip  # Set gripper joints to specified value (0 for closed, 1 for open)

        self._franka_manager._franka.control_dofs_position(qpos)
        self.log_spatial_info()

    def _rotate_end_effector(self, axis, angle):
        current_pos = self._franka_manager.get_ee_pos().cpu().numpy()
        current_quat = self._franka_manager.get_ee_quat().cpu().numpy()
        rotation_quat = self._create_rotation_quaternion(axis, angle)

        # Combine current quaternion with rotation quaternion
        new_quat = self._combine_quaternions(current_quat, rotation_quat)

        # Solve IK for the new rotation
        qpos, ik_diff_err = self._franka_manager._franka.inverse_kinematics(
            link=self._franka_manager._end_effector,
            pos=current_pos,
            quat=new_quat,
            return_error=True
        )

        if self._verbose:
            print(f"qpos: {qpos} \nIK error: {ik_diff_err}")

        self._franka_manager._franka.control_dofs_position(qpos)
        # self._scene.step()
        self.log_spatial_info()

    def _create_rotation_quaternion(self, axis, angle):
        angle_rad = np.radians(angle)
        if axis == 'x':
            return np.array([np.cos(angle_rad/2), np.sin(angle_rad/2), 0, 0])
        elif axis == 'y':
            return np.array([np.cos(angle_rad/2), 0, np.sin(angle_rad/2), 0])
        elif axis == 'z':
            return np.array([np.cos(angle_rad/2), 0, 0, np.sin(angle_rad/2)])
        else:
            raise ValueError("Invalid axis. Use 'x', 'y', or 'z'.")

    def _combine_quaternions(self, q1, q2):
        """
        Combines two quaternions q1 and q2.
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

    def log_spatial_info(self):
        if self._verbose:
            pos = self._franka_manager.get_ee_pos().cpu().numpy()
            quat = self._franka_manager.get_ee_quat().cpu().numpy()
            print(f"End-effector position: {pos} \nEnd-effector rotation (quaternion): {quat}")

    def open_grip(self):
        # Open the gripper by setting both finger joints to 1.0
        joint_positions, _ = self._franka_manager.get_joints_and_gripper_pos()
        joint_positions = joint_positions.cpu().numpy()

        franka_act = np.zeros(9)
        franka_act[:7] = joint_positions
        franka_act[7:9] = 1.0  # Open gripper

        self._franka_manager.set_joints_and_gripper_pos(franka_act)
        self.log_spatial_info()

    def close_grip(self):
        # Close the gripper by setting both finger joints to 0.0
        joint_positions, _ = self._franka_manager.get_joints_and_gripper_pos()
        joint_positions = joint_positions.cpu().numpy()

        franka_act = np.zeros(9)
        franka_act[:7] = joint_positions
        franka_act[7:9] = 0.0  # Close gripper

        self._franka_manager.set_joints_and_gripper_pos(franka_act)
        self.log_spatial_info()
