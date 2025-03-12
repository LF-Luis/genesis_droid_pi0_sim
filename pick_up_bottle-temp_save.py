import time
import argparse

import cv2
import numpy as np
import torch
import genesis as gs
from openpi.shared import download

# #TODO: DEBUG -- run before "from openpi.policies import policy_config" for now
print("A")
# Prime CV
image = np.zeros((300, 300, 3), dtype=np.uint8)
image[:] = (0, 255, 0)
print("A.1")
cv2.imshow("Test Window 1", image)
print("B")
for _ in range(3):
    cv2.waitKey(1)
    time.sleep(1)
print("C")
time.sleep(1)
cv2.destroyAllWindows()
time.sleep(1)
print("Done priming CV")
print("D")

from openpi.policies import policy_config
from openpi.training import config as openpi_config

"""
POC: Script to pick up a bottle using Franka Panda robot arm in Genesis Sim, being driven by OpenPI model.

Current versions:
    Genesis:
    https://github.com/Genesis-Embodied-AI/Genesis/commit/5cc3d5606c3c1e08eb3c628957e76e8e8512ae13
    OpenPi:
    https://github.com/Physical-Intelligence/openpi/commit/92b10824421d6d810eb1e398330acd79dc7cd934
"""

"""
rsync -avz -e "ssh -i ~/.ssh/aws-us-east-1.pem" \
    "$PWD/pick_up_bottle.py" \
    ubuntu@ec2-54-88-23-96.compute-1.amazonaws.com:/home/ubuntu/Desktop/Genesis-main/openpi/pick_up_bottle.py

python pick_up_bottle.py
python pick_up_bottle.py --model fast
"""

print("will start")


parser = argparse.ArgumentParser(description="OpenPI + Genesis: Franka picks up a bottle")
parser.add_argument("--model", choices=["fast", "diffusion"], default="fast")
args = parser.parse_args()



if args.model == "fast":
    # Autoregressive π0-FAST-DROID model
    model_name = "pi0_fast_droid"
else:
    # Diffusion π0-DROID model
    model_name = "pi0_droid"

print(f"model_name: {model_name}")

# Load the OpenPI model configuration and download the checkpoint
pi_config = openpi_config.get_config(model_name)
checkpoint_dir = download.maybe_download(f"s3://openpi-assets/checkpoints/{model_name}")
print("Done downloading model.")

# Create the policy object from the config and checkpoint
policy = policy_config.create_trained_policy(pi_config, checkpoint_dir)
print(f"Loaded OpenPI model '{model_name}' successfully.")

# Initialize Genesis (use GPU backend if available for faster physics & rendering)
gs.init(backend=gs.gpu)

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
    vis_options=gs.options.VisOptions(show_cameras=True),  # (show_cameras could be True for debugging)
    renderer=gs.renderers.Rasterizer()  # use rasterizer for rendering images
)

# Add a ground plane and the Franka Panda robot to the scene
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 0, 0)))

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


# Build the scene to finalize loading of entities
scene.build()


print("LF_DEBUG: CV warmup 2 ----")
time.sleep(1)
cv2.waitKey(3000)
cv2.destroyAllWindows()
time.sleep(1)
print("LF_DEBUG: Done CV warmup 2 ----")


# Define the robot joint indices for control (7 arm joints + 2 finger joints)
jnt_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
             "finger_joint1", "finger_joint2"]
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

# Optionally, set PD gains for smoother control (using values from Genesis example)
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]), dofs_idx)
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]), dofs_idx)

# After adding the bottle, step the simulation a few times to let it drop onto the plane
for _ in range(50):
    scene.step()


####################################
####################################

print("LF_DEBUG: About to show debug cam again----")

for step in range(50):
    scene.step()  # advance the simulation
    # Update camera views
    ext_camera.render()
    wrist_camera.render()
    cv2.waitKey(1)  # allow OpenCV to process the draw events

print("LF_DEBUG: Done showing debug cam again----")


def rotate_vector(quat, vec):
    # Ensure inputs are tensors
    if not isinstance(quat, torch.Tensor):
        print(f"rotate_vector: quat {quat} is not Tensor!!!")
        quat = torch.tensor(quat, dtype=torch.float32)
    if not isinstance(vec, torch.Tensor):
        print(f"rotate_vector: vec {vec} is not Tensor!!!")
        vec = torch.tensor(vec, dtype=torch.float32)

    w, x, y, z = quat
    # Build the rotation matrix using torch.stack
    row1 = torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y])
    row2 = torch.stack([2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x])
    row3 = torch.stack([2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y])
    rot_mat = torch.stack([row1, row2, row3])
    return torch.matmul(rot_mat, vec)





# Get reference to the robot's end-effector link (hand) for camera attachment
wrist_link = franka.get_link("hand")  # 'panda_hand' is the wrist link in the MJCF model

# Define a text prompt for the model
task_prompt = "pick up the bottle"

print("Starting control loop. Close the viewer window or press Ctrl+C to stop.")
# Main control loop
done = False
try:
    while not done:
        # 1. Capture observations from cameras
        # Update wrist camera pose to follow the gripper's current pose

        hand_pos = wrist_link.get_pos()
        hand_quat = wrist_link.get_quat()

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

        # 2. Prepare OpenPI input and infer action
        # Get the current joint positions and gripper position
        joint_positions = franka.get_dofs_position(dofs_idx[:7])  # First 7 DOFs are the arm joints
        gripper_position = franka.get_dofs_position(dofs_idx[7:8])  # 8th DOF is the first finger joint

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

        obs = {
            "observation/exterior_image_1_left": img_ext,
            "observation/wrist_image_left": img_wrist,
            "observation/joint_position": joint_positions,
            "observation/gripper_position": gripper_position,
            "prompt": task_prompt
        }
        result = policy.infer(obs)

        # The policy outputs a dictionary; extract the actions
        action_chunk = result["actions"]
        # Convert to numpy array for processing
        if isinstance(action_chunk, np.ndarray):
            action = action_chunk
        else:
            try:
                action = np.array(action_chunk)
            except:
                action = np.array(action_chunk.detach().cpu())  # handle torch tensor if needed

        # Print debug information about action shape
        print(f"Debug: Action shape: {action.shape}, DOFs length: {len(dofs_idx)}")
        print(f"Debug: Action data: {action}")
        print(f"Debug: DOFs indices: {dofs_idx}")

        # If the model produced a sequence of actions, take the first action for this step
        if action.ndim > 1:
            action = action[0]
            print(f"Debug: After taking first action: {action.shape}")

        # The policy outputs 8 values (7 for arm joints, 1 for gripper)
        # But the robot has 9 DOFs (7 for arm joints, 2 for finger joints)
        # Handle this by duplicating the gripper action for both finger joints
        if action.shape[0] == 8 and len(dofs_idx) == 9:
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
        franka.control_dofs_position(action, dofs_idx)

        # 4. Step the simulation forward a few steps to execute the action
        control_steps = 5  # number of simulation sub-steps per action
        for i in range(control_steps):
            scene.step()
            # Continuously update wrist cam to follow the moving hand
            hand_pos = wrist_link.get_pos()
            hand_quat = wrist_link.get_quat()
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
                import cv2
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

