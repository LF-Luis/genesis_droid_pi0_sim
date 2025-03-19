import os
import sys

import cv2
import torch
import numpy as np
import genesis as gs
from openpi_client import image_tools
from openpi_client import websocket_client_policy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.perf_timer import perf_timer
from franka_manager import FrankaManager
from sim_utils.cam_debug import CamDebugLayout
from sim_utils.transformations import rotate_vector


"""
Script to pick up a bottle using Franka Panda robot arm in Genesis Sim, being driven by OpenPI model.
"""

# Setup Params
COMPILE_KERNELS = True  # Set False only for debugging scene layout


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
        sim_options=gs.options.SimOptions(dt=0.01),   # simulation time-step 10ms
        vis_options=gs.options.VisOptions(show_cameras=False),  # (show_cameras could be True for debugging)
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
    cam_res = (320, 180)  # resolution similar to OpenPI training data
    # External camera: static position (e.g., above and in front of the robot, looking down at table)
    ext_camera = scene.add_camera(
        res=cam_res,
        pos=(0.5, -0.8, 0.4),    # position slightly in front of robot and above
        lookat=(0.5, 0.0, 0.0),  # look at the bottle on the plane
        fov=60,
        GUI=True
    )
    # Wrist camera: will be attached to the robot's wrist (end-effector) by updating its pose each step
    wrist_camera = scene.add_camera(
        res=cam_res,
        pos=(0.0, 0.0, 0.0),    # initial pose (will be updated dynamically)
        lookat=(0.0, 0.0, 0.0),
        fov=70,
        GUI=True
    )

with perf_timer("build scene"):  # takes 17.520714 seconds
    # Build the scene to finalize loading of entities
    scene.build(
        compile_kernels = COMPILE_KERNELS,  # Set to False when debugging scene layout
    )



franka_manager.set_to_init_pos()



# try:
#     import IPython
#     IPython.embed()
# except KeyboardInterrupt:
#     print("Simulation interrupted by user.")
# except Exception as e:
#     print(f"An error occurred: {e}")
# finally:
#     import sys; sys.exit()




# Define a text prompt for the model
task_prompt = "pick up the bottle"




print("Starting control loop. Close the viewer window or press Ctrl+C to stop.")
# Main control loop
done = False
try:
    while not done:
        # 1. Capture observations from cameras
        # Update wrist camera pose to follow the gripper's current pose

        hand_pos = franka_manager.get_ee_pos()
        hand_quat = franka_manager.get_ee_quat()

        print(f">>>>>>>>>> hand_pos: {hand_pos} | hand_quat: {hand_quat}")

        # Compute camera position slightly behind the gripper and orientation looking forward from gripper
        offset_back = rotate_vector(hand_quat, torch.tensor([0, 0, -0.1], device="cuda"))   # 10cm behind the hand in its local frame
        offset_fwd  = rotate_vector(hand_quat, torch.tensor([0, 0,  0.2], device="cuda"))   # 20cm in front of the hand
        cam_pos = hand_pos + offset_back
        cam_target = hand_pos + offset_fwd

        cam_pos = cam_pos.cpu().numpy()
        cam_target = cam_target.cpu().numpy()

        wrist_camera.set_pose(pos=tuple(cam_pos), lookat=tuple(cam_target))
        # Render the camera images (RGB only)
        img_ext = ext_camera.render()        # shape: (H, W, 3) uint8
        img_wrist = wrist_camera.render()


        hand_pos = franka_manager.get_ee_pos()
        hand_quat = franka_manager.get_ee_quat()
        print(f"start wrist pos: {hand_pos} | quat: {hand_quat}")
        print(f"start wrist type pos: {type(hand_pos)} | quat: {type(hand_quat)}")

        def set_wrist(hand_pos, hand_quat):
            """
            Set the camera pose using the hand's position and orientation.
            Offsets the camera 10cm in front of the hand to avoid occlusion.

            hand_pos: numpy array or torch tensor [x, y, z]
            hand_quat: numpy array or torch tensor [w, x, y, z] (in wxyz format)
            """
            # Convert to numpy if they're torch tensors
            if torch.is_tensor(hand_pos):
                hand_pos = hand_pos.cpu().numpy()
            if torch.is_tensor(hand_quat):
                hand_quat = hand_quat.cpu().numpy()

            # Calculate position 10cm in front of the hand
            # We use the hand's orientation to determine what "forward" means
            forward_offset = gs.utils.geom.transform_by_quat(np.array([0, 0, 0.1]), hand_quat)
            camera_pos = hand_pos + forward_offset

            # Create transform matrix using the offset position and hand's orientation
            transform = gs.utils.geom.trans_quat_to_T(camera_pos, hand_quat)

            # Set the camera pose
            wrist_camera.set_pose(transform=transform)
            wrist_camera.render()

        set_wrist(hand_pos, hand_quat)
        w_d = CamDebugLayout(wrist_camera, verbose = True)

        # TODO: print cam pos and quat after every movement

        # enter IPython's interactive mode
        cv2.waitKey(1)
        import IPython
        IPython.embed()



        # 2. Prepare OpenPI input and infer action

        # First 7 DOFs are the arm joints
        # 8th DOF is the first finger joint
        joint_positions, gripper_position = franka_manager.get_revolute_radians()

        # Convert CUDA tensors to numpy arrays
        if torch.is_tensor(joint_positions):
            joint_positions = joint_positions.cpu().numpy()
        if torch.is_tensor(gripper_position):
            gripper_position = gripper_position.cpu().numpy()

        # Ensure images are in correct format (H, W, 3) uint8
        if isinstance(img_ext, torch.Tensor):
            img_ext = img_ext.cpu().numpy()
        if isinstance(img_wrist, torch.Tensor):
            img_wrist = img_wrist.cpu().numpy()

        # Handle case where images are returned as tuples
        if isinstance(img_ext, tuple):
            img_ext = img_ext[0]  # Extract the image data from the tuple
        if isinstance(img_wrist, tuple):
            img_wrist = img_wrist[0]  # Extract the image data from the tuple

        # Ensure images are uint8 and in range [0, 255]
        if img_ext.dtype != np.uint8:
            img_ext = (img_ext * 255).astype(np.uint8)
        if img_wrist.dtype != np.uint8:
            img_wrist = (img_wrist * 255).astype(np.uint8)

        # Ensure images are in (H, W, 3) format
        if img_ext.shape[-1] != 3:
            img_ext = np.transpose(img_ext, (1, 2, 0))
        if img_wrist.shape[-1] != 3:
            img_wrist = np.transpose(img_wrist, (1, 2, 0))

        # Resize images on the client side to minimize bandwidth, latency, and match training routines.
        # Always return images in uint8 format.
        # The typical resize_size for pre-trained pi0 models is 224.
        # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
        observation = {
            "observation/exterior_image_1_left": image_tools.convert_to_uint8(image_tools.resize_with_pad(img_ext, 224, 224)),
            "observation/wrist_image_left": image_tools.convert_to_uint8(image_tools.resize_with_pad(img_wrist, 224, 224)),
            "observation/joint_position": joint_positions,
            "observation/gripper_position": gripper_position,
            "prompt": task_prompt,
        }

        try:
            action_chunk = pi0_model_client.infer(observation)["actions"]
        except Exception as e:
            print(f"Error running inference. Error: {e}")
            raise

        # Convert to numpy array for processing
        if isinstance(action_chunk, np.ndarray):
            action = action_chunk
        else:
            try:
                action = np.array(action_chunk)
            except:
                action = np.array(action_chunk.detach().cpu())  # handle torch tensor if needed

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
            # Continuously update wrist cam to follow the moving hand
            hand_pos = franka_manager.get_ee_pos()
            hand_quat = franka_manager.get_ee_quat()
            offset_back = rotate_vector(hand_quat, torch.tensor([0, 0, -0.1], device="cuda"))
            offset_fwd  = rotate_vector(hand_quat, torch.tensor([0, 0,  0.2], device="cuda"))
            cam_pos = hand_pos + offset_back
            cam_target = hand_pos + offset_fwd

            cam_pos = cam_pos.cpu().numpy()
            cam_target = cam_target.cpu().numpy()

            wrist_camera.set_pose(pos=tuple(cam_pos), lookat=tuple(cam_target))

            ext_camera.render()
            wrist_camera.render()
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
