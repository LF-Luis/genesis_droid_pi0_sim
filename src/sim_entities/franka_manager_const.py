import math
from typing import Tuple

import torch
import numpy as np

from src.sim_utils.transformations import quaternion_multiply


"""
Constants for frank_manager.py
"""

""" ORIGinal values
MUJOCO_FILE = "xml/franka_emika_panda/panda.xml"
# BASE_POS = (0, 0, 0)
BASE_POS = [0., -0.8, 0.]
# Robot joints: 7 arm joints + 2 finger joints
JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2"
]
END_EFFECTOR_NAME = "hand"
# Links PD gains
PROPORTIONAL_GAINS = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
VELOCITY_GAINS = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
# Links force ranges
FORCE_RANGES_LOWER = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100])
FORCE_RANGES_UPPER = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])

# HOME_POS = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
HOME_POS = np.array([-3.9809e-05, -5.5633e-01,  6.7885e-04, -2.6390e+00, -1.9741e-03,
                     2.2822e+00,  7.8514e-01,  3.9988e-02,  4.0000e-02])  # manually tuned

HOME_POS_STEPS = 50  # Steps to wait for stabilization
"""


# MUJOCO_FILE = "/workspace/explorations/assets/panda_wt_robotiq_2f85/panda_wt_2f85.xml"
# MUJOCO_FILE = "/workspace/dev/assets/panda_wt_robotiq_2f85/panda_wt_2f85.xml"
MUJOCO_FILE = "/workspace/explorations/assets/panda_wt_robotiq_2f85/panda_wt_2f85.xml"

BASE_POS = [0., -0.8, 0.]
# BASE_POS = [0., 0, 0.]
# Robot joints: 7 arm joints + 1 gripper driver actuator-joint
# BUG: Should be able to actuate with just the "left_driver_joint" joint, but maybe the Mujoco file or Genesis parser is off -- need to use "right_driver_joint" as well.
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "left_driver_joint", "right_driver_joint"]
END_EFFECTOR_NAME = "base"
# Links PD gains
PROPORTIONAL_GAINS = np.array([400, 400, 400, 400, 400, 400, 400,   15,   15])
VELOCITY_GAINS     = np.array([ 80,  80,  80,  80,  80,  80,  80,    5,    5])
# Links force ranges
FORCE_RANGES_LOWER = np.array([-87, -87, -87, -87, -12, -12, -12, -120, -120])
FORCE_RANGES_UPPER = np.array([ 87,  87,  87,  87,  12,  12,  12,  120,  120])

rest_pose = [-0.13935425877571106, -0.020481698215007782, -0.05201413854956627, -2.0691256523132324, 0.05058913677930832, 2.0028650760650635, -0.9167874455451965, 0., 0.]
HOME_POS = rest_pose
# HOME_POS = np.array([
#     0.0,
#     -1 / 5 * np.pi,
#     0.0,
#     -4 / 5 * np.pi,
#     0.0,
#     3 / 5 * np.pi,
#     0.0,
#     0.0,
#     0.0,
# ])
HOME_POS_STEPS = 150  # Steps to wait for stabilization

# Cam config values
CAM_RES = (1280, 720)  # Zed mini at 60 fps (this is for the wrist cam)
CAM_FOV = 57.  # Vertical FOV for Zed mini: 57 degrees

def get_wrist_cam_offset(device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Translation and rotation offset of wrist camera w.r.t. end-effector to put the camera
    under the wrist.
    Numbers were manually tuned.
    This should be called only once during init.
    """
    print(f"get_wrist_cam_offset device: {device}")
    # Position offset from robotics EE
    pos_offset = torch.tensor([-0.11, 0.0, -0.04], device=device)

    # Rotation offset w.r.t. robotics EE
    # 180° rotation about x-axis: flip camera's z-axis.
    q_x = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)  # [cos(pi/2), sin(pi/2), 0, 0]
    # -90° rotation about z-axis.
    angle_z = (math.pi / 2) * -1
    q_z = torch.tensor([math.cos(angle_z/2), 0.0, 0.0, math.sin(angle_z/2)], device=device)
    # 20° rotation about y-axis.
    angle_y = math.radians(20)
    q_y = torch.tensor([math.cos(angle_y/2), 0.0, math.sin(angle_y/2), 0.0], device=device)
    # Combine rotations: apply q_x, then q_z, then q_y
    rot_offset = quaternion_multiply(q_z, q_x)
    rot_offset = quaternion_multiply(q_y, rot_offset)

    return (pos_offset, rot_offset)
