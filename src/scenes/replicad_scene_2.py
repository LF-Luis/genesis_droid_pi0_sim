import numpy as np
import genesis as gs



"""
Setup simple scene and cameras to run pi0_fast_droid on.
White floor with a couple of objects to be picked..
Prompt used: "pick up the bottle from the white floor below"
"""

# Manually tuned exterior_image cams extrinsics
EXT_CAM_1_T = np.array([[ 0.96725646,  0.12152847, -0.22281333,  0.06295104],
                        [-0.2499452,   0.30367189, -0.91940784, -0.59134819],
                        [-0.04407208,  0.94499429,  0.32410405,  0.55773207],
                        [ 0.,          0.,          0.,          1.        ]])
EXT_CAM_2_T = np.array([[-0.8132096,   0.15450917, -0.56108562,  0.1298112 ],
                        [-0.5742504,  -0.36955738,  0.73052297,  0.51779325],
                        [-0.09448083,  0.91627194,  0.38925456,  0.52241897],
                        [ 0.,          0.,          0.,          1.        ]])


def setup_cams(scene):
    # Set up cameras for "exterior_image_1_left" and "exterior_image_2_left" data
    # FIXME: temp, using wrist cameras intrinsics
    from src.sim_entities.franka_manager_const import CAM_FOV, CAM_RES

    ext_cam_1_left = scene.add_camera(
        res=CAM_RES,
        pos=[0, 0, 0],
        lookat=[0, 0, 0],
        fov=CAM_FOV,
        GUI=True
    )
    # ext_cam_2_left = scene.add_camera(
    #     res=CAM_RES,
    #     pos=[0, 0, 0],
    #     lookat=[0, 0, 0],
    #     fov=CAM_FOV,
    #     GUI=True
    # )

    return ext_cam_1_left # , ext_cam_2_left


def setup_scene():
    # Set up the simulation scene with a viewer
    # scene = gs.Scene(
    #     show_viewer=True,
    #     viewer_options=gs.options.ViewerOptions(
    #         res=(1280, 720),
    #         camera_pos=(0.8, -1.0, 0.5),
    #         camera_lookat=(0.5, 0.0, 0.2),
    #         camera_fov=60,
    #         max_FPS=60,
    #     ),
    #     show_FPS = False,  # Don't print live FPS
    #     sim_options=gs.options.SimOptions(dt=0.01, substeps=2),  # simulation time-step 10ms, Defaults to 1e-2
    #     rigid_options=gs.options.RigidOptions(
    #         # Key: Increase contact solver stability
    #         iterations=150,  # More iterations for better convergence
    #         tolerance=1e-6,  # Tighter tolerance
    #         contact_resolve_time=0.01,  # Faster contact resolution
    #         # use_contact_island=True,  # use contact island to speed up contact resolving
    #     ),
    #     vis_options=gs.options.VisOptions(show_cameras=False),  # show where cameras are and where they're facing
    #     renderer=gs.renderers.Rasterizer(),  # use rasterizer for rendering images
    #     # renderer = gs.renderers.RayTracer())
    # )
        # Set up the simulation scene with a viewer
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 720),
            camera_pos=(0.8, -1.0, 0.5),
            camera_lookat=(0.5, 0.0, 0.2),
            camera_fov=60,
            max_FPS=60,
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.implicitfast,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=200,           # more constraint iterations
            ls_iterations=50,
            tolerance=1e-6,           # tighter convergence
            contact_resolve_time=0.02 # MuJoCo time-constant; prevents spiky impulses
            # enable_self_collision=False is already the default (good)
        ),
        show_FPS = False,  # Don't print live FPS
        # sim_options=gs.options.SimOptions(dt=0.01),  # simulation time-step 10ms, Defaults to 1e-2
        sim_options=gs.options.SimOptions(
            dt=0.002,  # 2ms step, mainly for the gripper stability
            # dt=0.01,
            # substeps=20,
            requires_grad=False,
        ),
        vis_options=gs.options.VisOptions(show_cameras=False),  # show where cameras are and where they're facing
        renderer=gs.renderers.Rasterizer()  # use rasterizer for rendering images
    )


    from src.scenes.replicad_parse_scene import parse_into_scene
    parse_into_scene(scene)
    basket_vis, basket_col = "", ""

    # from src.scenes.replicad_parse_scene_single_obj import parse_into_scene
    # basket_vis, basket_col = parse_into_scene(scene)


    # Add a ground plan
    plane = scene.add_entity(gs.morphs.Plane())

    # Add bottle object to scene
    bottle = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.URDF(
            file="urdf/3763/mobility_vhacd.urdf",
            scale=0.09,
            # pos=(0.5, 0.0, 0.1),
            # pos=(0.4, -0.9, 0.035),
            pos=(0.7, -0.9, 0.038),
            euler=(0, 90, 0),
        ),
    )

    # # Add little water wheel to scene
    # fancy_wheel = scene.add_entity(
    #     morph=gs.morphs.URDF(
    #         file="urdf/wheel/fancy_wheel.urdf",
    #         pos=(0.5, 0.25, 0.054),
    #         euler=(0, 0, 90),
    #         scale=0.04,
    #         convexify=False,
    #         # fixed=True,
    #     ),
    # )

    return scene, bottle, "room_entity", basket_vis, basket_col
