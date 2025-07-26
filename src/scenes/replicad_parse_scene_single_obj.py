import os, json
import numpy as np
import genesis as gs

# Path to ReplicaCAD dataset root
DATASET_PATH = "/workspace/assets/ReplicaCAD"
scene_name = "apt_0"  # choose your scene (e.g., "apt_0", etc.)
scene_config_file = os.path.join(DATASET_PATH, "configs/scenes", f"{scene_name}.scene_instance.json")
with open(scene_config_file, 'r') as f:
    scene_data = json.load(f)

def habitat_to_genesis_transform(translation, rotation_quat):
    """Convert Habitat (Y-up) pose to Genesis (Z-up) pose."""
    # Rotation of +90° about X-axis (to raise former Y to Z):
    Rx90 = np.array([[1, 0, 0],
                     [0, 0, -1],
                     [0, 1,  0]])
    trans = np.array(translation, dtype=float)
    new_pos = Rx90.dot(trans)  # (x, y, z)_gen = (x, -z, y)_hab
    # Quaternion (w, x, y, z) apply Rx90:
    qx90 = np.array([0.70710678, 0.70710678, 0, 0])  # 90° about X
    q = np.array(rotation_quat, dtype=float)         # Habitat quaternion (w, x, y, z)
    # Quaternion multiply: new_q = qx90 * q  (apply X-90 first, then original rotation)
    w1,x1,y1,z1 = qx90
    w2,x2,y2,z2 = q
    new_quat = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)
    new_quat /= np.linalg.norm(new_quat)
    return tuple(new_pos), tuple(new_quat)



def parse_into_scene(scene: gs.Scene):

    """
    Using example configs/objects/frl_apartment_basket.object_config.json

    Schema definition: https://aihabitat.org/docs/habitat-sim/attributesJSON.html

    - Don't use asset_cv_decomp because it has multiple convex-hull sub-meshes that Genesis doesn't natively read as individual pieces.
    So instead Genesis will load it as a single mesh
        -> Maybe do use it if it's easier for Genesis to decompose that the original .glb
    """

    # From `configs/objects/frl_apartment_basket.object_config.json`

    asset = f"{DATASET_PATH}/objects/frl_apartment_basket.glb"
    asset_cv_decomp = f"{DATASET_PATH}/objects/convex/frl_apartment_basket_cv_decomp.glb"
    mass = 0.5
    COM = [0, 0, 0]  # center of mass?
    position = [0.0, 0.0, -0.02796272560954094]
    # scale = [0.07694476842880249, 0.1227625384926796, 0.09645946323871613]

    # From `configs/scenes/apt_0.scene_instance.json`

    # translation = [ -1.9956579525706273, 1.0839370509764081, 0.057981376432922185 ]
    translation = [0.5, 0.25, 0.054]
    rotation = [ 0.9846951961517334, -5.20254616276361e-07, 0.17428532242774963, 3.540688453540497e-07 ]

    # Load into Genesis Scene

    translation, rotation = habitat_to_genesis_transform(translation, rotation)

    friction = 0.5
    restitution = 0.0


    # Basic GLB collision -- this works
    # basket_col = scene.add_entity(
    #     morph=gs.morphs.Mesh(
    #         file=asset,
    #         pos=translation,
    #         quat=rotation,
    #         collision=True,  # Enable collision physics
    #         visualization=True,  # Show the mesh
    #         fixed=False,  # Allow movement
    #     ),
    #     material=gs.materials.Rigid(
    #         rho=1000,  # Density
    #         friction=0.5,  # Friction
    #         coup_restitution=0.3,  # Bounciness
    #     ),
    # )
    # basket_vis = "basket_vis"

    # # Visual mesh (detailed, no collision)
    # basket_vis = scene.add_entity(
    #     morph=gs.morphs.Mesh(
    #         file=asset,
    #         pos=translation,
    #         quat=rotation,
    #         collision=False,  # No collision
    #         visualization=True,
    #         decimate=False,  # Keep original detail
    #     ),
    # )
    # # Collision mesh (simplified, invisible)
    # basket_col = scene.add_entity(   # <<<< only this one shows up
    #     morph=gs.morphs.Mesh(
    #         file=asset,
    #         pos=translation,
    #         quat=rotation,
    #         collision=True,  # Enable collision
    #         visualization=True, #False,  # Invisible
    #         decimate=True,  # Simplify for performance
    #         decimate_face_num=200,  # Low face count
    #     ),
    #     material=gs.materials.Rigid(friction=0.5),
    # )


    basket_col = scene.add_entity(
        gs.morphs.Mesh(
            file=asset,
            pos=translation,
            quat=rotation,
            # scale=scale,
            visualization=True,  # Whether to show it in the sim viewer
            collision=True,
            fixed=False,         # can be True or False, and obj still has collision
            convexify=False,     # Don't convert to convex-hull, try to keep original shape as much as possible (and most objects in scene have some concavity)
            decimate=True,       # Simplify mesh for collision
            decompose_nonconvex=False,
        ),
        # material=gs.materials.Rigid(friction=friction, coup_restitution=restitution),
        surface=gs.surfaces.Default(vis_mode="collision")  # See collision mesh
        # surface=gs.surfaces.Default(vis_mode="visual")  # See "visual" mesh, even-though the collision mesh will look different
    )
    basket_vis = "basket_vis"

    return basket_vis, basket_col
