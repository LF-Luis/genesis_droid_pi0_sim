import torch


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion (scalar-first, shape (4,)) to a 3x3 rotation matrix.
    """
    device = q.device
    dtype = q.dtype
    w, x, y, z = q  # q is expected to be [w, x, y, z]
    # Create the rotation matrix on the same device and with the same dtype as q.
    R = torch.tensor([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ], device=device, dtype=dtype)
    return R

def get_camera_transform(
    wrist_position: torch.Tensor,
    wrist_quaternion: torch.Tensor,
    offset_translation: torch.Tensor = None,
    offset_quaternion: torch.Tensor = None
) -> torch.Tensor:
    """
    Calculate the camera's 4x4 transformation matrix given the wrist's position and orientation.

    Parameters:
    - wrist_position: torch.Tensor of shape (3,)
    - wrist_quaternion: torch.Tensor of shape (4,) with scalar first ([w, x, y, z])
    - offset_translation: torch.Tensor of shape (3,), translation offset of the camera relative to the wrist.
      Defaults to a translation of 0.1 units along the local x-axis.
    - offset_quaternion: (optional) torch.Tensor of shape (4,) representing additional rotation offset.

    Returns:
    - T_camera: torch.Tensor of shape (4, 4) representing the camera transform in world coordinates.
    """
    device = wrist_position.device
    dtype = wrist_position.dtype

    # Use a default offset translation if none is provided.
    if offset_translation is None:
        offset_translation = torch.tensor([0.1, 0.0, 0.0], device=device, dtype=dtype)
    else:
        # Ensure offset_translation is on the same device.
        offset_translation = offset_translation.to(device=device, dtype=dtype)

    # Convert wrist quaternion to rotation matrix.
    R_wrist = quaternion_to_rotation_matrix(wrist_quaternion)

    # Construct the wrist's homogeneous transformation matrix on the same device.
    T_wrist = torch.eye(4, device=device, dtype=dtype)
    T_wrist[:3, :3] = R_wrist
    T_wrist[:3, 3] = wrist_position

    # Create the offset transformation.
    T_offset = torch.eye(4, device=device, dtype=dtype)
    T_offset[:3, 3] = offset_translation

    # If a rotation offset is provided, include it.
    if offset_quaternion is not None:
        R_offset = quaternion_to_rotation_matrix(offset_quaternion)
        T_offset[:3, :3] = R_offset

    # The camera transformation matrix in world coordinates.
    T_camera = T_wrist @ T_offset
    return T_camera

def change_coordinate_frame(transform: torch.Tensor, new_frame_position: torch.Tensor, new_frame_quaternion: torch.Tensor = None) -> torch.Tensor:
    """
    Convert a transform from world coordinates to be relative to a new coordinate frame.

    Parameters:
    - transform: torch.Tensor of shape (4, 4) - the transform in world coordinates
    - new_frame_position: torch.Tensor of shape (3,) - position of the new coordinate frame in world coordinates
    - new_frame_quaternion: torch.Tensor of shape (4,) - orientation of the new coordinate frame in world coordinates (scalar-first [w, x, y, z])
      If None, assumes identity rotation (no rotation)

    Returns:
    - torch.Tensor of shape (4, 4) - the transform relative to the new coordinate frame
    """
    device = transform.device
    dtype = transform.dtype

    # Ensure inputs are on the same device and dtype
    new_frame_position = new_frame_position.to(device=device, dtype=dtype)

    # Create the new frame's transformation matrix
    T_new_frame = torch.eye(4, device=device, dtype=dtype)
    T_new_frame[:3, 3] = new_frame_position

    # If quaternion is provided, include rotation
    if new_frame_quaternion is not None:
        new_frame_quaternion = new_frame_quaternion.to(device=device, dtype=dtype)
        R_new_frame = quaternion_to_rotation_matrix(new_frame_quaternion)
        T_new_frame[:3, :3] = R_new_frame

    # Compute the inverse of the new frame transform
    # For a rigid body transform, inverse = [R^T, -R^T * t]
    R_new_frame = T_new_frame[:3, :3]
    t_new_frame = T_new_frame[:3, 3]

    T_new_frame_inv = torch.eye(4, device=device, dtype=dtype)
    T_new_frame_inv[:3, :3] = R_new_frame.T
    T_new_frame_inv[:3, 3] = -R_new_frame.T @ t_new_frame

    # Convert the transform to the new coordinate frame
    # T_relative = T_new_frame_inv @ T_world
    transform_relative = T_new_frame_inv @ transform

    return transform_relative

def change_position_reference(transform: torch.Tensor, new_reference_position: torch.Tensor) -> torch.Tensor:
    """
    Change the position reference of a transform without affecting its orientation.
    This is useful when you want to keep the same orientation but change the coordinate frame origin.

    Parameters:
    - transform: torch.Tensor of shape (4, 4) - the transform in world coordinates
    - new_reference_position: torch.Tensor of shape (3,) - the new reference position in world coordinates

    Returns:
    - torch.Tensor of shape (4, 4) - the transform with position relative to new reference, but same orientation
    """
    device = transform.device
    dtype = transform.dtype

    # Ensure input is on the same device and dtype
    new_reference_position = new_reference_position.to(device=device, dtype=dtype)

    # Create a copy of the transform
    new_transform = transform.clone()

    # Only change the translation part (position) by subtracting the new reference position
    new_transform[:3, 3] = transform[:3, 3] - new_reference_position

    return new_transform

def move_relative_to_frame(transform: torch.Tensor, frame_position: torch.Tensor) -> torch.Tensor:
    """
    Move Transform to be positioned relative to a new frame, orientation doesn't change.

    Parameters:
    - transform: torch.Tensor of shape (4, 4) - original transform in world coordinates
    - frame_position: torch.Tensor of shape (3,) - the position of the new frame in world coordinates

    Returns:
    - torch.Tensor of shape (4, 4) - new transform positioned relative to the new frame
    """
    frame_position = frame_position.to(device=transform.device, dtype=transform.dtype)
    new_transform = transform.clone()
    new_transform[:3, 3] = transform[:3, 3] + frame_position
    return new_transform

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions (scalar-first).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.tensor([w, x, y, z], device=q1.device, dtype=q1.dtype)
