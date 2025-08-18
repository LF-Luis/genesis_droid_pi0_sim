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
    show_FPS = False,  # Don't print live FPS
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

# Load Panda with 2F85 gripper
panda = scene.add_entity(  # returns RigidEntity
    gs.morphs.MJCF(file="/workspace/dev/assets/panda_wt_robotiq_2f85/panda_wt_2f85.xml"),
    # gs.morphs.MJCF(file="/workspace/luis_dev/assets/panda_wt_robotiq_2f85/panda_wt_2f85-v2.xml"),
)
print([link.name for link in panda.links])
print([joint.name for joint in panda.joints])
"""
    ['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'base', 'left_driver', 'left_spring_link', 'right_driver', 'right_spring_link', 'left_coupler', 'left_follower', 'right_coupler', 'right_follower', 'left_pad', 'right_pad']
    ['link0_joint', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'base_joint', 'left_driver_joint', 'left_spring_link_joint', 'right_driver_joint', 'right_spring_link_joint', 'left_coupler_joint', 'left_follower_joint', 'right_coupler_joint', 'right_follower_joint', 'left_pad_joint', 'right_pad_joint']
"""


scene.build()

# Only joints that can, and should, be actuated
act_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "left_driver_joint"]
dofs_idx = [panda.get_joint(name).dof_idx_local for name in act_joints]
print(f"dofs_idx: {dofs_idx}")

# Set arm control gains
panda.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 50]),
    dofs_idx,
)
panda.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 5]),
    dofs_idx,
)
panda.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -5]),
    np.array([87, 87, 87, 87, 12, 12, 12, 5]),
    dofs_idx,
)

# Set Panda initial state
init_franka_pos = [1, 1, 0, 0, 0, 1, 0, 0]
panda.set_dofs_position(position=init_franka_pos, dofs_idx_local=dofs_idx, zero_velocity=True)
scene.reset(state=scene.get_state())

panda.control_dofs_position(
    np.array([
        0.0,
        -1 / 5 * np.pi,
        0.0,
        -4 / 5 * np.pi,
        0.0,
        3 / 5 * np.pi,
        0.0,
        0.0,
    ]),
    dofs_idx,
)

# # PD control
# for i in range(750):
#     if i == 0:
#         panda.control_dofs_position(
#             # np.array([1, 1, 0, 0, 0, 1, 0, 0]),
#             np.array([1, 1, 0, 0, 0, 0, 0, 0]),
#             dofs_idx,
#         )
#     elif i == 250:
#         panda.control_dofs_position(
#             np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0]),
#             dofs_idx,
#         )
#     elif i == 500:
#         panda.control_dofs_position(
#             # np.array([0, 0, 0, 0, 0, 1, 0, 0]),
#             np.array([0, 0, 0, 0, 0, 0, 0, 0]),
#             dofs_idx,
#         )
#     scene.step()


# enter_interactive(exit_at_end=True, stack_depth=1)  # ‼️ DEBUG MODE ‼️

while True:
    scene.step()