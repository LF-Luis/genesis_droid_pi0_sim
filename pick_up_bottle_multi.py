import traceback
import numpy as np
import genesis as gs
import torch
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from src.utils.perf_timer import perf_timer
from src.utils.debug import enter_interactive
from src.sim_utils.cam_pose_debug import CamPoseDebug
from src.scenes.simple_scene import setup_scene, setup_cams, EXT_CAM_1_T
from src.sim_utils.transformations import move_relative_to_frame
from src.sim_entities.franka_manager_const import (
    MUJOCO_FILE, BASE_POS, JOINT_NAMES, END_EFFECTOR_NAME, PROPORTIONAL_GAINS,
    VELOCITY_GAINS, FORCE_RANGES_LOWER, FORCE_RANGES_UPPER, HOME_POS, HOME_POS_STEPS,
    CAM_RES, CAM_FOV, get_wrist_cam_offset,
)


class FrankaManagerMulti:
    """
    Manage a Franka robot and all peripherals (e.g. cameras) attached to it in a given scene.
    MULTI-SIMULATION VERSION - properly handles batched environments.
    """

    def __init__(self, scene, n_envs=1):
        """
        Add robot to scene and set control config values.

        Args:
            scene: Genesis scene
            n_envs: Number of environments (default: 1 for single environment)
        """
        self._scene = scene
        self._n_envs = n_envs

        self._franka = self._scene.add_entity(
            gs.morphs.MJCF(file=MUJOCO_FILE, pos=BASE_POS),
            material=gs.materials.Rigid(
                friction=0.3,  # Lower friction for robot
                coup_restitution=0.0,  # No bouncing during coupling
            ),
            surface=gs.surfaces.Default(vis_mode="visual"),
        )

        self._end_effector = self._franka.get_link(name=END_EFFECTOR_NAME)
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
        print(f"Multi-simulation mode with {n_envs} environments")

    def _set_control_params(self):
        # Setting control parameters
        pass
        # self._franka.set_dofs_kp(PROPORTIONAL_GAINS, self.dofs_idx)
        # self._franka.set_dofs_kv(VELOCITY_GAINS, self.dofs_idx)
        # self._franka.set_dofs_force_range(FORCE_RANGES_LOWER, FORCE_RANGES_UPPER, self.dofs_idx)

    def set_to_init_pos(self):
        """
        Set control params and reset to an initial position for all environments.
        """
        from genesis.utils import geom as gu
        import numpy as np

        # Final tuned values for pos_offset and rot_offset
        pos_offset = np.array([0.03026469, 0.07047331, 0.02246456])
        rot_offset = np.array([-0.000662797508,  0.000376455792,  0.988124130,  0.153655398])  # (w,x,y,z)
        offset_T = gu.trans_quat_to_T(pos_offset, rot_offset)  # [qw, qx, qy, qz]

        self._wrist_camera.attach(rigid_link=self._end_effector, offset_T=offset_T)

        self._set_control_params()

        # Create batched home position for all environments
        batched_home_pos = np.tile(HOME_POS, (self._n_envs, 1))

        # Teleport to position - use the batched version
        self._franka.set_dofs_position(batched_home_pos, self.dofs_idx)
        # Solve for position - use the batched version
        self._franka.control_dofs_position(batched_home_pos, self.dofs_idx)

        # Wait for stabilization
        print(f"Running {HOME_POS_STEPS} steps to stabilize at home position.")
        for _ in range(HOME_POS_STEPS):
            self._scene.step()
        self.step()  # Perform setup once
        print(f"Done waiting for stabilization.")

    def get_ee_pos(self, env_idx=None) -> torch.Tensor:
        """
        Get end effector position.

        Args:
            env_idx: Environment index (None for all environments)
        """
        pos = self._end_effector.get_pos()
        if env_idx is not None and pos.ndim > 1:
            return pos[env_idx]
        return pos

    def get_ee_quat(self, env_idx=None) -> torch.Tensor:
        """
        Get end effector quaternion.

        Args:
            env_idx: Environment index (None for all environments)
        """
        quat = self._end_effector.get_quat()
        if env_idx is not None and quat.ndim > 1:
            return quat[env_idx]
        return quat

    def get_base_pos(self, env_idx=None) -> torch.Tensor:
        """
        Get robot base position.

        Args:
            env_idx: Environment index (None for all environments)
        """
        pos = self._franka.get_pos()
        if env_idx is not None and pos.ndim > 1:
            return pos[env_idx]
        return pos

    def get_base_quat(self, env_idx=None) -> torch.Tensor:
        """
        Get robot base quaternion.

        Args:
            env_idx: Environment index (None for all environments)
        """
        quat = self._franka.get_quat()
        if env_idx is not None and quat.ndim > 1:
            return quat[env_idx]
        return quat

    def get_joints_and_gripper_pos(self, env_idx=None):
        """
        Get the current joint and gripper revolute angles in radians.

        Args:
            env_idx: Environment index (None for all environments)

        Returns:
            joint_positions: Shape (7,) or (n_envs, 7)
            gripper_position: Shape (2,) or (n_envs, 2)
        """
        # Get the current joint and gripper revolute angles in radians
        dofs_positions = self._franka.get_dofs_position(dofs_idx_local=self.dofs_idx)  # 9 joints, held in CUDA Tensor

        # Handle single vs multi environment
        if dofs_positions.ndim == 1:
            # Single environment case
            joint_positions = dofs_positions[:7]  # First 7 DOFs are the arm joints
            gripper_position = dofs_positions[7:]  # 8th and 9th DOF is the gripper joints
        else:
            # Multi-environment case
            joint_positions = dofs_positions[:, :7]  # First 7 DOFs are the arm joints
            gripper_position = dofs_positions[:, 7:]  # 8th and 9th DOF is the gripper joints

            # Select specific environment if requested
            if env_idx is not None:
                joint_positions = joint_positions[env_idx]
                gripper_position = gripper_position[env_idx]

        return joint_positions, gripper_position

    def set_joints_and_gripper_pos(self, action, envs_idx=None):
        """
        Set the current joint and gripper positions.

        Args:
            action: Shape (9,) for single environment or (n_envs, 9) for multiple environments
            envs_idx: Specific environment indices to control (None for all)
        """
        if envs_idx is not None:
            # Control specific environments
            self._franka.control_dofs_position(action, self.dofs_idx, envs_idx=torch.tensor(envs_idx, device=gs.device))
        else:
            # Control all environments
            self._franka.control_dofs_position(action, self.dofs_idx)

    def cam_render(self, env_idx=None):
        """
        Render camera image.

        Args:
            env_idx: Environment index (None for all environments)
        """
        # Note: Camera rendering in multi-simulation might need special handling
        # For now, we'll render from the first environment or all environments
        return self._wrist_camera.render()

    def step(self, vis=True):
        """
        Run on each sim step (not tied to fps).
        Set vis = False to skip computing cam location and graphics during movement.
        """
        if vis:
            self._wrist_camera.move_to_attach()
            _ = self._wrist_camera.render()

    @property
    def franka(self):
        """Access to the underlying Franka entity."""
        return self._franka

    @property
    def n_envs(self):
        """Number of environments."""
        return self._n_envs


"""
Script to pick up a bottle using Franka Panda robot arm in Genesis Sim, being driven by OpenPI model.
MULTI-SIMULATION VERSION - runs multiple parallel environments with proper API handling.

DROID setup guidelines: https://droid-dataset.github.io/droid/docs/hardware-setup
Cameras info: https://www.stereolabs.com/store/products/zed-mini
"""

# Setup Params
COMPILE_KERNELS = True  # Set False only for debugging scene layout

SHOW_ROBOT = True
SHOW_SCENE_CAMS = True
RUN_PI0 = True

# Multi-simulation parameters
N_ENVIRONMENTS = 4  # Number of parallel environments to run
ENV_SPACING = (2.0, 2.0)  # Spacing between environments (x, y)

# Pi0 task prompt
task_prompt = "pick up the yellow bottle"
# task_prompt = "pick up the yellow bottle from white floor"
# task_prompt = "put the bottle in the bowl"

# Initialize link to OpenPi model, locally hosted
if RUN_PI0:
    pi0_model_client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

# Initialize Genesis
gs.init(
    backend=gs.gpu,
    logging_level="info",
    # performance_mode=True,  # compilation up to 6x slower (GJK), but runs ~1-5% faster
)

with perf_timer("Setup scene"):  # 1.24 seconds
    # scene, debug_bottle, debug_entity, basket_vis, basket_col = setup_scene()
    scene = setup_scene()

if SHOW_SCENE_CAMS:
    with perf_timer("Setup ext cams"):  # 0.000126 seconds
        ext_cam_1_left = setup_cams(scene)

if SHOW_ROBOT:
    with perf_timer("Setup Franka"):  # 1.08 secs
        franka_manager = FrankaManagerMulti(scene, n_envs=N_ENVIRONMENTS)

with perf_timer("Build scene"):  # 21.28 secs
    # Build the scene to finalize loading of entities with multiple environments
    scene.build(compile_kernels=COMPILE_KERNELS, n_envs=N_ENVIRONMENTS, env_spacing=ENV_SPACING)

if SHOW_SCENE_CAMS:
    # Correct placement
    from genesis.utils import geom as gu
    from src.sim_entities.franka_manager_const import BASE_POS

    cam_1_pos = np.array([0.05, 0.57, 0.66])
    cam_1_quat = np.array([-0.393, -0.195, 0.399, 0.805])  # (w,x,y,z)
    cam_1_T = gu.trans_quat_to_T(cam_1_pos, cam_1_quat)  # [qw, qx, qy, qz]

    # Attach left cam to base of robot
    ext_cam_1_left.attach(rigid_link=franka_manager.franka.base_link, offset_T=cam_1_T)


if SHOW_ROBOT:
    franka_manager.set_to_init_pos()

# Temp debug function to step through sim
def steps(n=1):
    for _ in range(n):
        scene.step()
        if SHOW_ROBOT:
            franka_manager.step()
        if SHOW_SCENE_CAMS:
            _ = ext_cam_1_left.render()

print(f"Starting simulation with {N_ENVIRONMENTS} parallel environments.")

steps(5)

if SHOW_ROBOT:
    from src.sim_utils.robot_pose_debug import RobotPoseDebug
    rD = RobotPoseDebug(franka_manager, scene, verbose=True)

if not RUN_PI0:
    enter_interactive(exit_at_end=True, stack_depth=1)

loop_step = -1
done = False
try:
    while not done:
        loop_step += 1  # Increase logical loop-step (not tied to actual sim step)

        if loop_step % 10 == 0 and loop_step > 0:
            enter_interactive()

        # Get scene "observation" data for Pi0 model (joint angles and cam images)
        # For multi-simulation, we'll focus on the first environment for now
        # In a full implementation, you'd want to handle all environments
        env_idx = 0  # Focus on first environment for demonstration

        ext_camera_img = ext_cam_1_left.render()[0]  # 0th is the rgb_arr
        wrist_cam_img = franka_manager.cam_render()[0]  # numpy.ndarray, uint8, Shape: (720, 1280, 3)

        # First 7 DOFs are the arm joints, 8th and 9th are the gripper
        # Now properly handles multi-environment data
        joint_positions, gripper_position = franka_manager.get_joints_and_gripper_pos(env_idx=env_idx)

        joint_positions = joint_positions.cpu().numpy()
        gripper_position = gripper_position.cpu().numpy()

        # For multi-simulation, joint_positions shape: (7,) for single env, (N_ENVIRONMENTS, 7) for all envs
        # Since we specified env_idx=0, we get shape (7,)

        """
        See src/data_inpection/droid_data.py
        'observation': FeaturesDict({
            'cartesian_position': Tensor(shape=(6,), dtype=float64),
            'exterior_image_1_left': Image(shape=(180, 320, 3), dtype=uint8),
            'exterior_image_2_left': Image(shape=(180, 320, 3), dtype=uint8),
            'gripper_position': Tensor(shape=(1,), dtype=float64),
            'joint_position': Tensor(shape=(7,), dtype=float64),
            'wrist_image_left': Image(shape=(180, 320, 3), dtype=uint8),
        }),
        """
        # Resize images on the client side to minimize bandwidth, latency, and match training routines.
        # Resizing it to 224x224 (as seen in openpi repo)
        ext_camera_img = image_tools.resize_with_pad(ext_camera_img, 224, 224)
        wrist_cam_img = image_tools.resize_with_pad(wrist_cam_img, 224, 224)

        gripper = gripper_position[0]
        gripper_norm = np.clip(gripper / (np.pi/4), 0.0, 1.0)
        gripper_norm = np.array([gripper_norm], dtype=np.float32)
        print(f"LF_DEBUG: gripper_position: {gripper_position}, gripper_norm: {gripper_norm}")  # Must end up between 0 and 1

        observation = {
            "observation/exterior_image_1_left": ext_camera_img,
            "observation/wrist_image_left": wrist_cam_img,
            "observation/joint_position": joint_positions,
            "observation/gripper_position": gripper_norm,  # must be a single number since it applies to both gripper fingers
            "prompt": task_prompt,
        }

        try:
            print(f"loop_step: {loop_step} | Running inference.")
            with perf_timer("Pi0 Inference Time"):
                model_response = pi0_model_client.infer(observation)
            actions = model_response["actions"][:8]  # LF_DEBUG only using the first 8 actions
        except Exception as e:
            # Pi0 server can fail when it's first started up, so try again by continuing to the next step
            print(f"⚠️ loop_step: {loop_step} | Error running inference. Will try again. Error: {e}")
            continue
        print(f"loop_step: {loop_step} | Running inference successful.")

        chuck_act_start_time = scene.cur_t
        import time
        chuck_start_wall_time = time.perf_counter()

        print(f"loop_step: {loop_step}. Sim time: {chuck_act_start_time:.3f} secs | Applying inference actions.")

        for i, action in enumerate(actions):
            arm_targets = action[:7]    # desired joint positions (radians)
            gripper_cmd = action[7]     # this will be 0.0 or 1.0 from model output
            print(f"LF_DEBUG: model out, gripper_cmd: {gripper_cmd}")

            # Map gripper command to actual joint value:
            if gripper_cmd > 0.2:
                gripper_target = np.pi / 4  # ~45° in radians, closed
            else:
                gripper_target = 0.0       # open
            franka_act = np.hstack([arm_targets, [gripper_target], [gripper_target]])

            # For multi-simulation, we need to apply actions to all environments
            # Create batched action for all environments
            batched_action = np.tile(franka_act, (N_ENVIRONMENTS, 1))
            franka_manager.set_joints_and_gripper_pos(batched_action)

            act_start_time = scene.cur_t
            steps(33)  # LF_DEBUG run at about 15 hz
            act_end_time = scene.cur_t
            act_diff_time = act_end_time - act_start_time

            print(f"loop_step: {loop_step} | Applying action {i+1}/{len(actions)}, took {act_diff_time:.3f} secs to run.")

        chuck_end_wall_time = time.perf_counter()
        chuck_diff_wall_time = chuck_end_wall_time - chuck_start_wall_time
        chuck_act_end_time = scene.cur_t
        chuck_act_diff_time = chuck_act_end_time - chuck_act_start_time
        print(f"loop_step: {loop_step} | Done applying inference actions, took {chuck_act_diff_time:.3f} secs in sim time, {chuck_diff_wall_time:.3f} secs wall clock time")

except KeyboardInterrupt:
    print("Simulation interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()
finally:
    print("Exiting simulation.")
