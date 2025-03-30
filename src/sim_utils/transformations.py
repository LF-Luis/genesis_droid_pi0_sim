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
