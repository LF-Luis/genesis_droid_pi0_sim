import torch


# FIXME: Remove type check -- convert to an assert
def rotate_vector(quat, vec):
    # Ensure inputs are tensors
    if not isinstance(quat, torch.Tensor):
        print(f"WARN: rotate_vector: quat {quat} is not Tensor!!!")
        quat = torch.tensor(quat, dtype=torch.float32)
    if not isinstance(vec, torch.Tensor):
        print(f"WARN: rotate_vector: vec {vec} is not Tensor!!!")
        vec = torch.tensor(vec, dtype=torch.float32)

    w, x, y, z = quat
    # Build the rotation matrix using torch.stack
    row1 = torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y])
    row2 = torch.stack([2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x])
    row3 = torch.stack([2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y])
    rot_mat = torch.stack([row1, row2, row3])
    return torch.matmul(rot_mat, vec)
