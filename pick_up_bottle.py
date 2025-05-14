import cv2
import torch
import numpy as np
import genesis as gs
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from src.utils.perf_timer import perf_timer
from src.sim_entities.franka_manager import FrankaManager
from src.sim_utils.cam_pose_debug import CamPoseDebug


"""
Script to pick up a bottle using Franka Panda robot arm in Genesis Sim, being driven by OpenPI model.

DROID setup guidelines: https://droid-dataset.github.io/droid/docs/hardware-setup
Cameras info: https://www.stereolabs.com/store/products/zed-mini
"""

# Setup Params
COMPILE_KERNELS = True  # Set False only for debugging scene layout

# Pi0 task prompt
task_prompt = "pick up the yellow bottle from the blue white below"

# Initialize link to OpenPi model, locally hosted
pi0_model_client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

# Initialize Genesis
gs.init(backend=gs.gpu)

with perf_timer("setup scene"):  # takes 29.097971 seconds
    # Set up the simulation scene with a viewer
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 720),
            camera_pos=(0.8, -1.0, 0.5),   # position the viewer camera behind and above the robot
            camera_lookat=(0.5, 0.0, 0.2), # look at the area where the bottle is expected
            camera_fov=60,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),  # simulation time-step 10ms, Defaults to 1e-2
        vis_options=gs.options.VisOptions(show_cameras=False),  # show where cameras are and where they're facing
        renderer=gs.renderers.Rasterizer()  # use rasterizer for rendering images
    )

with perf_timer("add items to scene"):
    # Add a ground plane and the Franka Panda robot to the scene
    white_surface = gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0))
    plane = scene.add_entity(gs.morphs.Plane(), surface=white_surface)
    franka_manager = FrankaManager(scene)

    # Add the bottle object to the scene
    bottle = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.URDF(
            file="urdf/3763/mobility_vhacd.urdf",
            scale=0.09,
            pos=(0.5, 0.0, 0.1),     # place bottle in front of robot, slightly above ground
            # pos=(0.65, 0.0, 0.036),
            euler=(0, 90, 0),
        ),
    )

    # Set up cameras for external view and wrist view
    # FIXME: temp, using wrist cameras intrinsics
    from src.sim_entities.franka_manager_const import CAM_FOV, CAM_RES
    # External camera: static position, looking at (part of) the robot, manipulator, and scene with object

    # Manually tuned
    ext_cam_T = np.array([[ 9.08489575e-01,  2.00108195e-01, -3.66883365e-01, -1.41301081e-01],
                          [-4.17907517e-01,  4.35015408e-01, -7.97568118e-01, -5.60818185e-01],
                          [ 5.27355937e-16,  8.77905636e-01,  4.78833681e-01, 6.97813210e-01],
                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
    ext_camera = scene.add_camera(
        res=CAM_RES,
        pos=[0, 0, 0],
        # pos=[-0.14130108, -0.56081819, 0.69781321],
        lookat=[0, 0, 0],
        # lookat=[0.15467354, 0.26569094, 0.21897953],
        fov=CAM_FOV,
        GUI=True
    )

    print(f"Intrinsics: \nexternal cam: \n{ext_camera.intrinsics}")

with perf_timer("build scene"):  # takes 17.520714 seconds
    # Build the scene to finalize loading of entities
    scene.build(
        compile_kernels = COMPILE_KERNELS,  # Set to False when debugging scene layout
    )
    ext_camera.set_pose(transform=ext_cam_T)


franka_manager.set_to_init_pos()


# Temp debug function to step through sim
def steps(n=10):
    for _ in range(n):
        scene.step()
        franka_manager.step()
        _ = ext_camera.render()


def inspect_structure(obj):
    print("=== Structure Info ===")

    # Outer type
    print(f"Outer type: {type(obj)}")

    # Number of elements
    try:
        length = len(obj)
        print(f"Number of top-level elements: {length}")
    except TypeError:
        print("Object has no length (not iterable)")
        return

    # Data type of elements
    if isinstance(obj, np.ndarray):
        print(f"Numpy dtype: {obj.dtype}")
        print(f"Shape: {obj.shape}")
        print(f"Inner type (inferred from first element): {type(obj.flat[0]) if obj.size > 0 else 'N/A'}")

    elif torch and isinstance(obj, torch.Tensor):
        print(f"PyTorch dtype: {obj.dtype}")
        print(f"Shape: {obj.shape}")
        print(f"Inner type (inferred): {type(obj.flatten()[0].item()) if obj.numel() > 0 else 'N/A'}")

    elif isinstance(obj, list):
        if len(obj) > 0:
            first_type = type(obj[0])
            print(f"Inner type (first element): {first_type}")
            try:
                inner_value_type = type(obj[0][0])
                print(f"Nested value type (first element of first element): {inner_value_type}")
            except Exception:
                pass
        else:
            print("List is empty, inner type unknown.")
    else:
        print("Unsupported type or unknown structure.")

print("Starting simulation.")


# from sim_utils.robot_pose_debug import RobotPoseDebug
# rD = RobotPoseDebug(franka_manager, scene, True)
# cD = CamPoseDebug(camera=ext_camera, verbose=True)

steps(50)

step_num = 0
done = False
try:
    while not done:

        # Get scene "observation" data for Pi0 model (joint angles and cam images)
        ext_camera_img = ext_camera.render()[0]  # 0th is the rgb_arr
        wrist_cam_img = franka_manager.cam_render()[0]  # numpy.ndarray, uint8, Shape: (720, 1280, 3)

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
        # debug
        from genesis.utils.tools import save_img_arr
        save_img_arr(ext_camera_img, "ext_camera_img_pre_resize.png")
        save_img_arr(wrist_cam_img, "wrist_cam_img_pre_resize.png")

        ext_camera_img = image_tools.resize_with_pad(ext_camera_img, 224, 224)
        wrist_cam_img = image_tools.resize_with_pad(wrist_cam_img, 224, 224)

        # debug
        from genesis.utils.tools import save_img_arr
        save_img_arr(ext_camera_img, "ext_camera_img.png")
        save_img_arr(wrist_cam_img, "wrist_cam_img.png")

        gripper_position = np.array([gripper_position[0]])

        observation = {
            "observation/exterior_image_1_left": ext_camera_img,
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
        print("action_chunk:")
        # inspect_structure(actions)
        print(actions)

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
            Running at 1/20 = 0.05 secs = 50 ms per action
            Each step is 10ms
            So 5 steps per action
            """
            # TODO: make gripper binary, 0.5 threshold
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
                franka_act[7:9] = 0.75  # 1.0

            # print(f"final franka_act: {franka_act}")
            franka_manager.set_joints_and_gripper_pos(franka_act)

            steps(20)

        if step_num % 20 == 0:
            # Enter IPython's interactive mode
            import IPython
            IPython.embed()
            # import sys; sys.exit()

        # Increase step count
        step_num += 1


except KeyboardInterrupt:
    print("Simulation interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Exiting simulation.")
