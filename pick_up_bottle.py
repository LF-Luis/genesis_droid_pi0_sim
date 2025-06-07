import numpy as np
import genesis as gs
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from src.utils.perf_timer import perf_timer
# from src.utils.debug import inspect_structure
from src.sim_utils.cam_pose_debug import CamPoseDebug
from src.sim_entities.franka_manager import FrankaManager
from src.scenes.simple_scene import setup_scene, setup_cams, EXT_CAM_1_T, EXT_CAM_2_T


"""
Script to pick up a bottle using Franka Panda robot arm in Genesis Sim, being driven by OpenPI model.

DROID setup guidelines: https://droid-dataset.github.io/droid/docs/hardware-setup
Cameras info: https://www.stereolabs.com/store/products/zed-mini
"""

# Setup Params
COMPILE_KERNELS = True  # Set False only for debugging scene layout

# Pi0 task prompt
# task_prompt = "pick up the yellow bottle from the white floor below"
task_prompt = "pick up the bottle from the white floor below"

# Initialize link to OpenPi model, locally hosted
# pi0_model_client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

# Initialize Genesis
gs.init(backend=gs.gpu)

with perf_timer("Setup scene"):  # 1.24 seconds
    scene = setup_scene()

# with perf_timer("Setup ext cams"):  # 0.000126 seconds
#     ext_cam_1_left, ext_cam_2_left = setup_cams(scene)

with perf_timer("Setup Franka"):  # 1.08 secs
    franka_manager = FrankaManager(scene)

with perf_timer("Build scene"):  # 21.28 secs
    # Build the scene to finalize loading of entities
    scene.build(compile_kernels = COMPILE_KERNELS)

# ext_cam_1_left.set_pose(transform=EXT_CAM_1_T)
# ext_cam_2_left.set_pose(transform=EXT_CAM_2_T)
franka_manager.set_to_init_pos()

# Temp debug function to step through sim
def steps(n=10):
    for _ in range(n):
        scene.step()
        franka_manager.step()
        # _ = ext_cam_1_left.render()
        # _ = ext_cam_2_left.render()

print("Starting simulation.")

steps(3)

from src.sim_utils.robot_pose_debug import RobotPoseDebug

rD = RobotPoseDebug(franka_manager, scene, verbose=True)

## >> DEBUG
import IPython
IPython.embed()
import sys; sys.exit()
## <<

step_num = 0
done = False
try:
    while not done:

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

        print(f"observation: {observation}")

        try:
            model_response = pi0_model_client.infer(observation)
            actions = model_response["actions"]  # Shape: (10, 8), numpy.float64
        except Exception as e:
            print(f"Error running inference. Error: {e}")
            raise


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
        for action in actions:
            """
            Running at 20 Hz = 1/20 secs = 0.05 secs = 50 ms per action
            Each step is 10ms
            So 5 steps per action
            """
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

        # if step_num > 0 and step_num % 50 == 0:
        #     # Enter IPython's interactive mode
        #     import IPython
        #     IPython.embed()
        #     # import sys; sys.exit()

        # Increase step count
        step_num += 1


except KeyboardInterrupt:
    print("Simulation interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Exiting simulation.")
