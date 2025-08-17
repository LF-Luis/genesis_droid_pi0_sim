import numpy as np
import genesis as gs
from src.utils.debug import enter_interactive
from src.scenes.simple_scene import setup_scene

gs.init(backend=gs.gpu, logging_level="info")
scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 720),
        camera_pos=(0.8, -1.0, 0.5),
        camera_lookat=(0.5, 0.0, 0.2),
        camera_fov=60,
        max_FPS=60,
    ),
    # ↓↓↓ main fixes ↓↓↓
    sim_options=gs.options.SimOptions(dt=0.002),  # 2 ms step like Menagerie scenes
    rigid_options=gs.options.RigidOptions(
        integrator=gs.integrator.implicitfast,
        constraint_solver=gs.constraint_solver.Newton,
        iterations=200,           # more constraint iterations
        ls_iterations=50,
        tolerance=1e-6,           # tighter convergence
        contact_resolve_time=0.02 # ~MuJoCo time-constant; prevents spiky impulses
        # enable_self_collision=False is already the default (good)
    ),
    vis_options=gs.options.VisOptions(show_cameras=False),
    renderer=gs.renderers.Rasterizer()
)
# Add a ground plane
white_surface = gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0))
plane = scene.add_entity(gs.morphs.Plane(), surface=white_surface)
# Add bottle object to scene
bottle = scene.add_entity(
    material=gs.materials.Rigid(rho=300),
    morph=gs.morphs.URDF(
        file="urdf/3763/mobility_vhacd.urdf",
        scale=0.09,
        # pos=(0.5, 0.0, 0.1),
        pos=(0.5, -0.25, 0.0343),
        euler=(0, 90, 0),
    ),
)


# Load 2F85 gripper
gripper_path = "/workspace/luis_dev/assets/robotiq_2f85_v4/mjx_2f85.xml"
# gripper_path = "xml/franka_emika_panda/hand.xml"
gripper = scene.add_entity(  # returns RigidEntity
    gs.morphs.MJCF(file=gripper_path),
)
print([link.name for link in gripper.links])
print([joint.name for joint in gripper.joints])
"""
['base', 'left_driver', 'left_spring_link', 'right_driver', 'right_spring_link', 'left_coupler', 'left_follower', 'right_coupler', 'right_follower', 'left_pad', 'right_pad']
['base_joint', 'left_driver_joint', 'left_spring_link_joint', 'right_driver_joint', 'right_spring_link_joint', 'left_coupler_joint', 'left_follower', 'right_coupler_joint', 'right_follower_joint', 'left_pad_joint', 'right_pad_joint']
"""

scene.build()

# Only joints that can, and should, be actuated
# act_joints = ["left_driver_joint"]
# dofs_idx = [gripper.get_joint(name).dof_idx_local for name in act_joints]
# print(f"dofs_idx: {dofs_idx}")

# Set arm control gains
# gripper.set_dofs_kp(
#     np.array([50]),
#     dofs_idx,
# )
# gripper.set_dofs_kv(
#     np.array([5]),
#     dofs_idx,
# )
# gripper.set_dofs_force_range(
#     np.array([-5]),
#     np.array([5]),
#     dofs_idx,
# )

def step(n=1):
    for _ in range(n):
        scene.step()

enter_interactive(exit_at_end=True, stack_depth=1)  # ‼️ DEBUG MODE ‼️

while True:
    scene.step()