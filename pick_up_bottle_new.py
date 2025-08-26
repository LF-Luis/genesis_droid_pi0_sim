import traceback

import numpy as np
import genesis as gs
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from src.utils.perf_timer import perf_timer
from src.utils.debug import enter_interactive #, inspect_structure
from src.sim_utils.cam_pose_debug import CamPoseDebug
from src.sim_entities.franka_manager import FrankaManager
from src.scenes.simple_scene import setup_scene, setup_cams, EXT_CAM_1_T


from src.sim_utils.transformations import move_relative_to_frame
import torch

"""
Script to pick up a bottle using Franka Panda robot arm in Genesis Sim, being driven by OpenPI model.

DROID setup guidelines: https://droid-dataset.github.io/droid/docs/hardware-setup
Cameras info: https://www.stereolabs.com/store/products/zed-mini
"""

# Setup Params
COMPILE_KERNELS = True  # Set False only for debugging scene layout

SHOW_ROBOT = True
SHOW_SCENE_CAMS = True
RUN_PI0 = True

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
        franka_manager = FrankaManager(scene)

with perf_timer("Build scene"):  # 21.28 secs
    # Build the scene to finalize loading of entities
    scene.build(compile_kernels = COMPILE_KERNELS)

if SHOW_SCENE_CAMS:
    # Move camera transforms to be positioned relative to robot base, but keep world orientation
    # robot_base_pos = franka_manager.get_base_pos()
    # EXT_CAM_1_T_robot_frame = move_relative_to_frame(
    #     torch.tensor(EXT_CAM_1_T, device=gs.device),
    #     robot_base_pos
    # )
    # # Set camera poses using the converted transforms
    # ext_cam_1_left.set_pose(transform=EXT_CAM_1_T_robot_frame.cpu().numpy())

    # Correct placement
    from genesis.utils import geom as gu
    from src.sim_entities.franka_manager_const import BASE_POS

    cam_1_pos = np.array([0.05, 0.57, 0.66])
    cam_1_quat = np.array([-0.393, -0.195, 0.399, 0.805])  # (w,x,y,z)
    cam_1_T = gu.trans_quat_to_T(cam_1_pos, cam_1_quat)  # [qw, qx, qy, qz]
    # ext_cam_1_left.set_pose(transform=cam_1_T)

    # Attach left cam to base of robot
    ext_cam_1_left.attach(rigid_link=franka_manager._franka.base_link, offset_T=cam_1_T)


if SHOW_ROBOT:
    franka_manager.set_to_init_pos()

# Temp debug function to step through sim
# def steps(n=10):
def steps(n=1):
    for _ in range(n):
        scene.step()
        if SHOW_ROBOT:
            franka_manager.step()
        if SHOW_SCENE_CAMS:
            _ = ext_cam_1_left.render()

print("Starting simulation.")

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
        ext_camera_img = ext_cam_1_left.render()[0]  # 0th is the rgb_arr
        wrist_cam_img = franka_manager.cam_render()[0]  # numpy.ndarray, uint8, Shape: (720, 1280, 3)


        # # debug
        # from genesis.utils.tools import save_img_arr
        # save_img_arr(ext_camera_img, "ext_camera_img_pre_resize.png")
        # save_img_arr(wrist_cam_img, "wrist_cam_img_pre_resize.png")


        # First 7 DOFs are the arm joints, 8th and 9th are the gripper
        joint_positions, gripper_position = franka_manager.get_joints_and_gripper_pos()  # CUDA torch.float32

        joint_positions = joint_positions.cpu().numpy()
        gripper_position = gripper_position.cpu().numpy()

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

        # # debug
        # from genesis.utils.tools import save_img_arr
        # save_img_arr(ext_camera_img, "ext_camera_img.png")
        # save_img_arr(wrist_cam_img, "wrist_cam_img.png")

        gripper = gripper_position[0]
        gripper_norm = np.clip(gripper / (np.pi/4), 0.0, 1.0)
        gripper_norm = np.array([gripper_norm], dtype=np.float32)
        # gripper_position = np.array([gripper_position[0]])
        print(f"LF_DEBUG: gripper_position: {gripper_position}, gripper_norm: {gripper_norm}")  # Must end up between 0 and 1

        observation = {
            "observation/exterior_image_1_left": ext_camera_img,
            "observation/wrist_image_left": wrist_cam_img,
            "observation/joint_position": joint_positions,
            # "observation/gripper_position": gripper_position,  # must be a single number since it applies to both gripper fingers
            # "observation/gripper_position": np.array([gripper_position[0]]),  # must be a single number since it applies to both gripper fingers
            "observation/gripper_position": gripper_norm,  # must be a single number since it applies to both gripper fingers
            "prompt": task_prompt,
        }

        # print(f"observation: {observation}")

        try:
            print(f"loop_step: {loop_step} | Running inference.")
            with perf_timer("Pi0 Inference Time"):
                model_response = pi0_model_client.infer(observation)
            # actions = model_response["actions"]  # Shape: (10, 8), numpy.float64
            actions = model_response["actions"][:8]  # LF_DEBUG only using the first 8 actions
        except Exception as e:
            # Pi0 server can fail when it's first started up, so try again by continuing to the next step
            print(f"⚠️ loop_step: {loop_step} | Error running inference. Will try again. Error: {e}")
            # cleanup_and_recreate_pi0_client()
            continue
            # raise
        print(f"loop_step: {loop_step} | Running inference successful.")

        # Debug
        # print("action_chunk:")
        # inspect_structure(actions)
        # print(actions)

        """
        At every inference call, π0 outputs a 10x8 chunk. That is 10 actions, with each action having 8
        values that correspond to the 8 joints+gripper in the Franka robot (7 rotary joints, 1 gripper action).
        Note that Franka defined in has 9 joints+gripper states (7 rotary joints, 2 gripper action). The 1 gripper
        action from π0 is applied to both gripper fingers.

        In the π0 paper, Franka runs at 20Hz with a 16 step horizon (π0 paper, APPENDIX D. Inference). Here I'll
        use 20Hz as well, but with a 10 step horizon per chunk. We'll record the tracking error (q_desired - q_actual).
        """

        """
        DEBUG: run each action at full step completion to make sure model output makes sense, then run at 20 Hz only
        """

        chuck_act_start_time = scene.cur_t
        import time
        chuck_start_wall_time = time.perf_counter()

        print(f"loop_step: {loop_step}. Sim time: {chuck_act_start_time:.3f} secs | Applying inference actions.")

        for i, action in enumerate(actions):
            """
            when running at dt=0.01
            Running Pi0 at 20 Hz (per paper) = 1/20 secs = 0.05 secs = 50 ms per action
            Each step is 10ms
            So 5 steps per action
            """
            """
            when running at dt=0.002
            Running Pi0 at 20 Hz (per paper) = 1/20 secs = 0.05 secs = 50 ms per action
            Each step is 2ms
            (50 ms per action) / (2ms per step) = 25
            So 25 steps per action
            """
            """
            when running at dt=0.002
            Running sim-evals-Pi0 at 15 Hz (per repo) = 1/15 secs = 0.0667 secs = 66 ms per action
            Each step is 2ms
            (66 ms per action) / (2ms per step) = 33.3
            So 33.3 steps per action
            """

            # print(f"loop_step: {loop_step} | Applying action {i+1}/{len(actions)}")
            # print(f"start joint_positions: {joint_positions}")
            # print(f"start gripper_position: {gripper_position}")
            # print(f"action: {action}")

            ############################################
            ######### Delta joint pos approach #########
            # # Extract arm joint actions (first 7 elements) and gripper action (8th element)
            # arm_action = action[:7]  # Shape: (7,)
            # gripper_action = action[7]  # Shape: (1,)

            # # Apply delta to arm joints
            # franka_act = joint_positions + arm_action  # Shape: (7,)
            # # franka_act = arm_action  # LF_DEBUG -- using velocity actions

            # # Handle gripper separately - convert to absolute position
            # if gripper_action > 0.5:
            #     gripper_target = np.pi / 4  # ~45° in radians, closed
            # else:
            #     gripper_target = 0.0  # open

            # # Combine arm joints and gripper (duplicate gripper target for both fingers)
            # franka_act = np.hstack([franka_act, [gripper_target], [gripper_target]])  # Shape: (9,)
            # # print(f"final franka_act: {franka_act}")
            # franka_manager.set_joints_and_gripper_pos(franka_act)
            ############################################
            ############################################

            ###############################################
            ######### Absolute joint pos approach #########
            arm_targets = action[:7]    # desired joint positions (radians)
            gripper_cmd = action[7]     # this will be 0.0 or 1.0 from model output
            print(f"LF_DEBUG: model out, gripper_cmd: {gripper_cmd}")
            # Map gripper command to actual joint value:
            # if gripper_cmd > 0.5:
            if gripper_cmd > 0.2:
                gripper_target = np.pi / 4  # ~45° in radians, closed
            else:
                gripper_target = 0.0       # open
            franka_act = np.hstack([arm_targets, [gripper_target], [gripper_target]])
            franka_manager.set_joints_and_gripper_pos(franka_act)
            ###############################################
            ###############################################

            act_start_time = scene.cur_t
            # steps(5)
            # steps(25)
            # steps(10)  # LF_DEBUG run at about 50 hz
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
