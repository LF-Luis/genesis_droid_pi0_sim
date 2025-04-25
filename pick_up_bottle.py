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
task_prompt = "pick up the yellow bottle from the blue surface below"

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
        sim_options=gs.options.SimOptions(dt=0.01),  # simulation time-step 10ms
        vis_options=gs.options.VisOptions(show_cameras=False),  # show where cameras are and where they're facing
        renderer=gs.renderers.Rasterizer()  # use rasterizer for rendering images
    )

with perf_timer("add items to scene"):
    # Add a ground plane and the Franka Panda robot to the scene
    plane = scene.add_entity(gs.morphs.Plane())
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

done = False
try:
    while not done:

        from sim_utils.robot_pose_debug import RobotPoseDebug
        rD = RobotPoseDebug(franka_manager, scene, True)
        cD = CamPoseDebug(camera=ext_camera, verbose=True)

        steps(50)

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
        observation = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(ext_camera_img, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(wrist_cam_img, 224, 224),
            "observation/joint_position": joint_positions,
            "observation/gripper_position": np.array([gripper_position[0]]),  # must be a single number since it applies to both gripper fingers
            "prompt": task_prompt,
        }

        print(f"observation: {observation}")

        try:
            model_response = pi0_model_client.infer(observation)
            action = model_response["actions"]  # Shape: (10, 8), numpy.float64
        except Exception as e:
            print(f"Error running inference. Error: {e}")
            raise


        print("action_chunk:"); inspect_structure(action)

        # Enter IPython's interactive mode
        import IPython
        IPython.embed()
        import sys; sys.exit()

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


        # Print debug information about action shape
        print(f"Debug: Action shape: {action.shape}, DOFs length: {len(franka_manager.dofs_idx)}")
        print(f"Debug: Action data: {action}")
        print(f"Debug: DOFs indices: {franka_manager.dofs_idx}")

        # If the model produced a sequence of actions, take the first action for this step
        if action.ndim > 1:
            action = action[0]
            print(f"Debug: After taking first action: {action.shape}")

        # The policy outputs 8 values (7 for arm joints, 1 for gripper)
        # But the robot has 9 DOFs (7 for arm joints, 2 for finger joints)
        # Handle this by duplicating the gripper action for both finger joints
        if action.shape[0] == 8 and len(franka_manager.dofs_idx) == 9:
            arm_action = action[:7]  # First 7 values for arm joints
            gripper_action = action[7]  # 8th value for gripper

            # Create a 9-dimensional action with the gripper action duplicated
            full_action = np.zeros(9)
            full_action[:7] = arm_action  # Copy arm actions
            full_action[7:9] = gripper_action  # Same gripper action for both fingers

            action = full_action
            print(f"Debug: Expanded action shape: {action.shape}")
            print(f"Debug: Expanded action data: {action}")

        action = action.astype(float)

        # 3. Apply the action to the robot (position control for each DoF)
        franka_manager.set_revolute_radians(action)

        # 4. Step the simulation forward a few steps to execute the action
        control_steps = 5  # number of simulation sub-steps per action
        for i in range(control_steps):
            scene.step()

            ext_camera.render()

            # Small delay to update display
            try:
                cv2.waitKey(1)
            except:
                pass

        # 5. Check if the bottle is lifted (success condition)
        bottle_pos = bottle.get_pos()  # get bottle's center position
        if bottle_pos[2] > 0.2:  # if the bottle's height > 20cm (adjust threshold as needed)
            print("Bottle picked up successfully!")
            done = True

except KeyboardInterrupt:
    print("Simulation interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Exiting simulation.")
