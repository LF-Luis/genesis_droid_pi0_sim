import numpy as np
import genesis as gs
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from src.utils.perf_timer import perf_timer
from src.utils.debug import enter_interactive #, inspect_structure
from src.sim_utils.cam_pose_debug import CamPoseDebug
from src.sim_entities.franka_manager import FrankaManager
from src.scenes.simple_scene import setup_scene, setup_cams, EXT_CAM_1_T, EXT_CAM_2_T
# from src.scenes.replicad_scene import setup_scene, setup_cams, EXT_CAM_1_T, EXT_CAM_2_T
# from src.scenes.replicad_scene_2 import setup_scene, setup_cams, EXT_CAM_1_T, EXT_CAM_2_T


from src.sim_utils.transformations import move_relative_to_frame
import torch


import gc
def cleanup_and_recreate_pi0_client():
    """
    Completely garbage collect the pi0_model_client and create a new instance when client becomes unresponsive
    """
    global pi0_model_client
    # Delete the reference
    if 'pi0_model_client' in globals():
        del pi0_model_client
    # Force garbage collection
    gc.collect()
    # Create a new instance
    pi0_model_client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
    print("Successfully recreated pi0_model_client")
    return pi0_model_client


"""
Script to pick up a bottle using Franka Panda robot arm in Genesis Sim, being driven by OpenPI model.

DROID setup guidelines: https://droid-dataset.github.io/droid/docs/hardware-setup
Cameras info: https://www.stereolabs.com/store/products/zed-mini
"""

# Setup Params
COMPILE_KERNELS = True  # Set False only for debugging scene layout

SHOW_ROBOT = False
SHOW_SCENE_CAMS = False  # True
RUN_PI0 = False  # True

# Pi0 task prompt
# task_prompt = "pick up the yellow bottle from the white floor below"
# task_prompt = "pick up the bottle from the white floor below"
task_prompt = "pick up the yellow bottle"
# task_prompt = "grab the leg of the table"

# Initialize link to OpenPi model, locally hosted
if RUN_PI0:
    pi0_model_client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

# Initialize Genesis
gs.init(
    backend=gs.gpu,
    logging_level="info",
)

with perf_timer("Setup scene"):  # 1.24 seconds
    # scene, debug_bottle, debug_entity, basket_vis, basket_col = setup_scene()
    scene = setup_scene()

if SHOW_SCENE_CAMS:
    with perf_timer("Setup ext cams"):  # 0.000126 seconds
        ext_cam_1_left, ext_cam_2_left = setup_cams(scene)

if SHOW_ROBOT:
    with perf_timer("Setup Franka"):  # 1.08 secs
        franka_manager = FrankaManager(scene)

# Load Frank with no EEF
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda_nohand.xml"),
)

# Load default Franka EEF
# hand = scene.add_entity(
#     gs.morphs.MJCF(file="xml/franka_emika_panda/hand.xml"),
# )
# """
# ['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'attachment']
# ['hand', 'left_finger', 'right_finger']
# """
# scene.link_entities(franka, hand, "attachment", "hand")

# Load default Robotiq-2F85 EEF
# gripper_path = "/workspace/assets/robotiq_2f85_v4/mjx_2f85.xml"
"""
['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'attachment']
['link0_joint', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'attachment_joint']
['base', 'left_driver', 'left_spring_link', 'right_driver', 'right_spring_link', 'left_coupler', 'left_follower', 'right_coupler', 'right_follower', 'left_pad', 'right_pad']
['base_joint', 'left_driver_joint', 'left_spring_link_joint', 'right_driver_joint', 'right_spring_link_joint', 'left_coupler_joint', 'left_follower', 'right_coupler_joint', 'right_follower_joint', 'left_pad_joint', 'right_pad_joint']
"""
gripper_path = "/workspace/assets/robotiq_2f85_v4/2f85.xml"
"""
['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'attachment']
['base', 'left_driver', 'left_spring_link', 'right_driver', 'right_spring_link', 'left_coupler', 'left_follower', 'right_coupler', 'right_follower', 'left_pad', 'right_pad']
"""
# gripper_path = "/workspace/assets/robotiq_2f85/2f85.xml"
"""
['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'attachment']
['base_mount', 'base', 'right_driver', 'right_spring_link', 'left_driver', 'left_spring_link', 'right_coupler', 'right_follower', 'left_coupler', 'left_follower', 'right_pad', 'left_pad', 'right_silicone_pad', 'left_silicone_pad']
"""

hand = scene.add_entity(
    gs.morphs.MJCF(file=gripper_path),
)
# print(f'type(franka): {type(franka)}')  # RigidEntity
# print(f'type(hand): {type(hand)}')  # RigidEntity
print([link.name for link in franka.links])
print([joint.name for joint in franka.joints])
print([link.name for link in hand.links])
print([joint.name for joint in hand.joints])
scene.link_entities(franka, hand, "attachment", "base")
# scene.link_entities(franka, hand, "attachment", "base_mount")



with perf_timer("Build scene"):  # 21.28 secs
    # Build the scene to finalize loading of entities
    scene.build(compile_kernels = COMPILE_KERNELS)


arm_joints_name = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
arm_dofs_idx = [franka.get_joint(name).dof_idx_local for name in arm_joints_name]
print(f"arm_dofs_idx: {arm_dofs_idx}")  # [0, 1, 2, 3, 4, 5, 6]

# Set arm control gains
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
    arm_dofs_idx,
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200]),
    arm_dofs_idx,
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12]),
    np.array([87, 87, 87, 87, 12, 12, 12]),
    arm_dofs_idx,
)

gripper_joints_name = ['base_joint', 'left_driver_joint', 'left_spring_link_joint', 'right_driver_joint', 'right_spring_link_joint', 'left_coupler_joint', 'left_follower', 'right_coupler_joint', 'right_follower_joint', 'left_pad_joint', 'right_pad_joint']
# gripper_dofs_idx = [hand.get_joint(name).dof_idx_local for name in gripper_joints_name]
# print(f"gripper_dofs_idx: {gripper_dofs_idx}")  # [None, 0, 1, 2, 3, None, 4, None, 5, None, None]
# gripper_dofs_idx = [0, 1, 2, 3, 4, 5]
# gripper_dofs_idx = [hand.get_joint("left_driver_joint").dof_idx_local, hand.get_joint("right_driver_joint").dof_idx_local]
gripper_dofs_idx = [hand.get_joint("left_driver_joint").dof_idx_local]
print(f"gripper_dofs_idx: {gripper_dofs_idx}")

# # Set gripper control gains
hand.set_dofs_kp(np.array([50.0]), gripper_dofs_idx)
hand.set_dofs_kv(np.array([5.0 ]), gripper_dofs_idx)
hand.set_dofs_force_range(np.array([-5.0]), np.array([5.0]), gripper_dofs_idx)
# hand.set_dofs_kp(
#     np.array([100] * len(gripper_dofs_idx)),
#     gripper_dofs_idx,
# )
# hand.set_dofs_kv(
#     np.array([10] * len(gripper_dofs_idx)),
#     gripper_dofs_idx,
# )
# hand.set_dofs_force_range(
#     np.array([-100] * len(gripper_dofs_idx)),
#     np.array([100] * len(gripper_dofs_idx)),
#     gripper_dofs_idx,
# )

# Set Franka initial state
init_franka_pos = [1, 1, 0, 0, 0, 1, 0]
franka.set_dofs_position(position=init_franka_pos, zero_velocity=True)
scene.reset(state=scene.get_state())

# PD control
for i in range(750):
    if i == 0:
        franka.control_dofs_position(
            np.array([1, 1, 0, 0, 0, 1, 0]),
            arm_dofs_idx,
        )
        """
franka.control_dofs_position(
    np.array([1, 1, 0, 0, 0, 0, 1]),
    arm_dofs_idx,
)
scene.step()
        """
        hand.control_dofs_position(np.array([0.0]), gripper_dofs_idx)
        # hand.control_dofs_position(
        #     np.array([0.0] * len(gripper_dofs_idx)),
        #     gripper_dofs_idx,
        # )
    elif i == 250:
        franka.control_dofs_position(
            np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5]),
            arm_dofs_idx,
        )
        hand.control_dofs_position(
            np.array([0.0] * len(gripper_dofs_idx)),
            gripper_dofs_idx,
        )
    elif i == 500:
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 1, 0]),
            arm_dofs_idx,
        )
        hand.control_dofs_position(
            np.array([0.0] * len(gripper_dofs_idx)),
            gripper_dofs_idx,
        )

    scene.step()
    enter_interactive()




if SHOW_SCENE_CAMS:
    # Move camera transforms to be positioned relative to robot base, but keep world orientation
    robot_base_pos = franka_manager.get_base_pos()
    EXT_CAM_1_T_robot_frame = move_relative_to_frame(
        torch.tensor(EXT_CAM_1_T, device=gs.device),
        robot_base_pos
    )
    EXT_CAM_2_T_robot_frame = move_relative_to_frame(
        torch.tensor(EXT_CAM_2_T, device=gs.device),
        robot_base_pos
    )
    # Set camera poses using the converted transforms
    ext_cam_1_left.set_pose(transform=EXT_CAM_1_T_robot_frame.cpu().numpy())
    ext_cam_2_left.set_pose(transform=EXT_CAM_2_T_robot_frame.cpu().numpy())

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
            _ = ext_cam_2_left.render()

print("Starting simulation.")

steps(10)

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

        if loop_step % 10 == 0:
            enter_interactive()

        # Get scene "observation" data for Pi0 model (joint angles and cam images)
        ext_camera_img = ext_cam_1_left.render()[0]  # 0th is the rgb_arr
        ext_camera_img_2 = ext_cam_2_left.render()[0]  # 0th is the rgb_arr
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
        ext_camera_img_2 = image_tools.resize_with_pad(ext_camera_img_2, 224, 224)
        wrist_cam_img = image_tools.resize_with_pad(wrist_cam_img, 224, 224)

        # # debug
        # from genesis.utils.tools import save_img_arr
        # save_img_arr(ext_camera_img, "ext_camera_img.png")
        # save_img_arr(wrist_cam_img, "wrist_cam_img.png")

        gripper_position = np.array([gripper_position[0]])

        observation = {
            "observation/exterior_image_1_left": ext_camera_img,
            "observation/exterior_image_2_left": ext_camera_img_2,
            "observation/wrist_image_left": wrist_cam_img,
            "observation/joint_position": joint_positions,
            "observation/gripper_position": gripper_position,  # must be a single number since it applies to both gripper fingers
            "prompt": task_prompt,
        }

        # print(f"observation: {observation}")

        try:
            print(f"loop_step: {loop_step} | Running inference.")
            with perf_timer("Pi0 Inference Time"):
                model_response = pi0_model_client.infer(observation)
            actions = model_response["actions"]  # Shape: (10, 8), numpy.float64
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
        print(f"loop_step: {loop_step} | Applying inference actions.")
        for i, action in enumerate(actions):
            """
            Running at 20 Hz = 1/20 secs = 0.05 secs = 50 ms per action
            Each step is 10ms
            So 5 steps per action
            """
            print(f"loop_step: {loop_step} | Applying action {i+1}/{len(actions)}")
            # print(f"start joint_positions: {joint_positions}")
            # print(f"start gripper_position: {gripper_position}")
            # print(f"action: {action}")

            # Gripper is considered open when the action value is greater than 0.5

            franka_act = np.zeros(9)
            franka_act[:7] = joint_positions + action[:7]
            # franka_act[7:9] = gripper_position + action[7]

            # Open/close is opposite in Genesis URDF of Franka?
            if action[-1].item() > 0.5:
                franka_act[7:9] = 0.0
            else:
                franka_act[7:9] = 0.6  # 1.0

            # print(f"final franka_act: {franka_act}")
            franka_manager.set_joints_and_gripper_pos(franka_act)

            steps(5)

        print(f"loop_step: {loop_step} | Done applying inference actions.")

except KeyboardInterrupt:
    print("Simulation interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Exiting simulation.")
