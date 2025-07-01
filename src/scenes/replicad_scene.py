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
    ext_cam_2_left = scene.add_camera(
        res=CAM_RES,
        pos=[0, 0, 0],
        lookat=[0, 0, 0],
        fov=CAM_FOV,
        GUI=True
    )

    return ext_cam_1_left, ext_cam_2_left


def setup_scene():
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
        show_FPS = False,  # Don't print live FPS
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2),  # simulation time-step 10ms, Defaults to 1e-2
        rigid_options=gs.options.RigidOptions(
            # Key: Increase contact solver stability
            iterations=150,  # More iterations for better convergence
            tolerance=1e-6,  # Tighter tolerance
            contact_resolve_time=0.01,  # Faster contact resolution
        ),
        vis_options=gs.options.VisOptions(show_cameras=False),  # show where cameras are and where they're facing
        renderer=gs.renderers.Rasterizer()  # use rasterizer for rendering images
    )

    # Add a ground plan
    plane = scene.add_entity(gs.morphs.Plane())

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

    # Add little water wheel to scene
    fancy_wheel = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/wheel/fancy_wheel.urdf",
            pos=(0.5, 0.25, 0.054),
            euler=(0, 0, 90),
            scale=0.04,
            convexify=False,
            # fixed=True,
        ),
    )

    ############################################################################################################

    # MUG_FILE_PATH = "/workspace/genesis/assets/urdf/ACE_Coffee_Mug_Kristen_16_oz_cup/model.xml"
    # mug = scene.add_entity(
    #     gs.morphs.MJCF(
    #         file=MUG_FILE_PATH,
    #         pos=(0.5, 0.0, 0.054),
    #     )
    # )

    ############################################################################################################

    # stage_path = "/workspace/assets/Baked_sc2_staging_00.glb"
    # morph = gs.morphs.Mesh(
    #     file=stage_path,
    #     group_by_material=True,  # Group meshes by material, speeds up parsing
    #     requires_jac_and_IK=False,  # Static scene doesn't require Jacobian or IK
    #     # pos=(0, 0.5, -0.015),
    #     pos=(4, 0.5, -0.05),
    #     euler=(90,0,0),  # euler angle of the entity in degrees, follows scipy's extrinsic x-y-z rotation convention
    #     fixed=True,
    # )
    # # Add the mesh to the scene
    # entity = scene.add_entity(
    #     morph=morph
    # )

    # Load with Trimesh
    # morph = gs.morphs.Mesh(
    #     file=stage_path,
    #     group_by_material=True,  # Group meshes by material, speeds up parsing
    #     decimate=True,  # Simplify mesh
    #     convexify=True,  # Convert meshes to convex hull approx.
    #     merge_submeshes_for_collision=True,  # Reduce number of collision objects
    #     requires_jac_and_IK=False,  # Static scene doesn't require Jacobian or IK
    #     euler=(90,0,0),  # euler angle of the entity in degrees, follows scipy's extrinsic x-y-z rotation convention
    #     parse_glb_with_trimesh=True,  # Doesn't load GLB properly
    #     fixed=True,
    # )
    # entity = scene.add_entity(
    #     morph=morph
    # )

    ############################################################################################################

    # # Load room with SIMPLIFIED collision geometry
    # stage_path = "/workspace/assets/Baked_sc2_staging_00.glb"

    # room_morph = gs.morphs.Mesh(
    #     file=stage_path,
    #     group_by_material=True,
    #     requires_jac_and_IK=False,
    #     pos=(0, 0.5, -0.05),
    #     euler=(90, 0, 0),
    #     fixed=True,
    #     # CRITICAL: Simplify collision geometry to prevent numerical issues
    #     collision=True,
    #     convexify=True,  # Force convex shapes
    #     decompose_nonconvex=True,  # Break into convex pieces
    #     decimate=True,
    #     decimate_face_num=200,  # Very low face count for collision
    #     merge_submeshes_for_collision=True,
    # )

    # # Add room with stable material properties
    # room_entity = scene.add_entity(
    #     morph=room_morph,
    #     material=gs.materials.Rigid(
    #         friction=0.5,  # Moderate friction
    #         coup_restitution=0.0,  # No bouncing during coupling
    #     ),
    #     surface=gs.surfaces.Default(vis_mode="visual"),
    # )

    ############################################################################################################

    stage_path = "/workspace/assets/Baked_sc2_staging_00.glb"

    # Visual room mesh (no collisions)
    room_visual = gs.morphs.Mesh(
        file=stage_path,
        pos=(0, 0.5, -0.05),    # position/offset to align floor
        euler=(90, 0, 0),       # rotate from Y-up to Z-up coordinates
        fixed=True,
        collision=False,        # visual only, no collision geometry
        requires_jac_and_IK=False,
        group_by_material=True  # preserve materials for rendering
    )
    scene.add_entity(morph=room_visual, surface=gs.surfaces.Default(vis_mode="visual"))

    # Collision room mesh (invisible, simplified)
    room_collision = gs.morphs.Mesh(
        file=stage_path,
        pos=(0, 0.5, -0.05),
        euler=(90, 0, 0),
        fixed=True,
        # visualization=False,    # do not visualize this mesh
        visualization=True,
        collision=True,         # enable collisions for this mesh
        convexify=False,        # don't collapse into one convex hull (preserve shape)
        decompose_nonconvex=True, # decompose into multiple convex pieces for concave areas
        decimate=True,
        decimate_face_num=500,  # simplify mesh to ~500 faces for collision stability
        merge_submeshes_for_collision=True   # combine sub-meshes before decomposition
    )
    room_entity = scene.add_entity(
        morph=room_collision,
        material=gs.materials.Rigid(friction=0.5, coup_restitution=0.0),  # static material properties
        # No surface needed (visualization=False makes it invisible)
    )


    # return scene
    return scene, bottle, room_entity
