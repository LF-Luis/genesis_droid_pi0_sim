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

def determine_urdf_path(name: str) -> str:
    """Determine URDF file path for articulated object by its template name."""
    urdf_base = os.path.join(DATASET_PATH, "urdf")
    if name.startswith("chestOfDrawers"):
        return os.path.join(urdf_base, "chest_of_drawers", f"{name}.urdf")
    elif name.startswith("kitchenCupboard"):
        return os.path.join(urdf_base, "kitchen_cupboards", f"{name}.urdf")
    elif name.startswith("door"):
        return os.path.join(urdf_base, "doors", f"{name}.urdf")
    else:
        # e.g. "fridge", "cabinet", etc.
        return os.path.join(urdf_base, name, f"{name}.urdf")

def parse_into_scene(scene: gs.Scene):

    # 1. Load the static stage (environment shell like walls/floor)
    stage_template = scene_data["stage_instance"]["template_name"]  # e.g. "stages/frl_apartment_stage"
    stage_name = os.path.basename(stage_template)                   # e.g. "frl_apartment_stage"
    stage_cfg_path = os.path.join(DATASET_PATH, "configs/stages", f"{stage_name}.stage_config.json")
    with open(stage_cfg_path, 'r') as f:
        stage_cfg = json.load(f)
    stage_render_asset = os.path.join(DATASET_PATH, "stages", os.path.basename(stage_cfg["render_asset"]))
    stage_collision_asset = stage_cfg.get("collision_asset")
    if stage_collision_asset:
        stage_collision_asset = os.path.join(DATASET_PATH, "stages", os.path.basename(stage_collision_asset))
    else:
        stage_collision_asset = stage_render_asset  # use render mesh if no separate collision mesh
    stage_scale = stage_cfg.get("units_to_meters", 1.0)
    friction = stage_cfg.get("friction_coefficient", 0.5)
    restitution = stage_cfg.get("restitution_coefficient", 0.0)
    # Add stage visual (Y-up -> Z-up: apply 90° X rotation)
    scene.add_entity(
        gs.morphs.Mesh(file=stage_render_asset, pos=(0,0,0), euler=(90,0,0), scale=stage_scale,
                       visualization=True, collision=False, fixed=True),
        surface=gs.surfaces.Default(vis_mode="visual")
    )
    # Add stage collision
    # scene.add_entity(
    #     gs.morphs.Mesh(file=stage_collision_asset, pos=(0,0,0), euler=(90,0,0), scale=stage_scale,
    #                    visualization=False, collision=True, fixed=True,
    #                    convexify=False, decimate=False, decompose_nonconvex=True),  # decompose large concave stage
    #     material=gs.materials.Rigid(friction=friction, coup_restitution=restitution),
    #     surface=gs.surfaces.Default(vis_mode="collision")
    # )


    # 2. Load static object instances (furniture, etc.)
    for obj in scene_data.get("object_instances", []):
        template = obj["template_name"]          # e.g. "objects/frl_apartment_table"
        obj_name = os.path.basename(template)    # e.g. "frl_apartment_table"
        obj_cfg_path = os.path.join(DATASET_PATH, "configs/objects", f"{obj_name}.object_config.json")
        if not os.path.exists(obj_cfg_path):
            print(f"Warning: no config for {obj_name}, skipping")
            continue
        with open(obj_cfg_path, 'r') as f:
            obj_cfg = json.load(f)
        # Paths for visual and collision meshes
        vis_asset = os.path.join(DATASET_PATH, "objects", os.path.basename(obj_cfg["render_asset"]))
        col_asset = obj_cfg.get("collision_asset")
        if col_asset:
            # collision_asset is relative to config file location
            col_asset = os.path.normpath(os.path.join(os.path.dirname(obj_cfg_path), col_asset))
        else:
            col_asset = vis_asset  # fallback to visual mesh if no separate collision mesh
        # Physical properties
        scale = obj_cfg.get("scale", 1.0) * obj_cfg.get("units_to_meters", 1.0)
        friction = obj_cfg.get("friction_coefficient", 0.5)
        restitution = obj_cfg.get("restitution_coefficient", 0.0)
        # Transform pose to Genesis frame
        pos_hab = obj.get("translation", [0,0,0])
        rot_hab = obj.get("rotation", [1,0,0,0])
        pos_gen, quat_gen = habitat_to_genesis_transform(pos_hab, rot_hab)

        motion_type = obj["motion_type"]
        fixed = True
        if motion_type == "DYNAMIC":
            fixed = False
        elif motion_type == "STATIC":
            fixed = True
        else:
            print(f"> Missing 'motion_type' entry for: {template}")

        # Add visual mesh (no collision)
        # scene.add_entity(
        #     gs.morphs.Mesh(file=vis_asset, pos=pos_gen, quat=quat_gen, scale=scale,
        #                    visualization=True, collision=False, fixed=True),
        #     surface=gs.surfaces.Default(vis_mode="visual")
        # )
        # Add entity with collision mesh, view visual mesh
        scene.add_entity(
            gs.morphs.Mesh(
                file=vis_asset,
                pos=pos_gen,
                quat=quat_gen,
                scale=scale,
                visualization=True,  # Whether to show it in the sim viewer
                collision=True,
                fixed=fixed,         # can be True or False, and obj still has collision
                convexify=False,     # Don't convert to convex-hull, try to keep original shape as much as possible (and most objects in scene have some concavity)
                decimate=True,       # Simplify mesh for collision
                decompose_nonconvex=False,
            ),
            # surface=gs.surfaces.Default(vis_mode="visual"),
            surface=gs.surfaces.Default(vis_mode="collision")
        )
        # scene.add_entity(
        #     gs.morphs.Mesh(file=vis_asset,  # col_asset,
        #                    pos=pos_gen, quat=quat_gen, scale=scale,
        #                    visualization=False, collision=True, fixed=True,
        #                    convexify=False, decimate=False, decompose_nonconvex=False,
        #                    group_by_material=False, merge_submeshes_for_collision=False),
        #     material=gs.materials.Rigid(friction=friction, coup_restitution=restitution),
        #     surface=gs.surfaces.Default(vis_mode="visual")
        #     # surface=gs.surfaces.Default(vis_mode="collision")
        # )


    # 3. Load articulated objects (doors, cabinets with URDFs)
    for art in scene_data.get("articulated_object_instances", []):
        name = art["template_name"]  # e.g. "fridge", "door1", "kitchenCabinet_01", ...
        urdf_path = determine_urdf_path(name)
        if not os.path.exists(urdf_path):
            print(f"(Skipping {name}, URDF not found: {urdf_path})")
            continue
        pos_hab = art.get("translation", [0,0,0])
        rot_hab = art.get("rotation", [1,0,0,0])
        pos_gen, quat_gen = habitat_to_genesis_transform(pos_hab, rot_hab)
        scene.add_entity(
            gs.morphs.URDF(file=urdf_path, pos=pos_gen, quat=quat_gen, fixed=art.get("fixed_base", True)),
            material=gs.materials.Rigid(friction=0.5, coup_restitution=0.0),
            surface=gs.surfaces.Default(vis_mode="visual")
        )
