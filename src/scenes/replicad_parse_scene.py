"""
Git LFS is needed
git clone https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset
"""

import json, os
import numpy as np
import genesis as gs

# Path to the downloaded ReplicaCAD dataset (Interactive version)
DATASET_PATH = "/workspace/assets/ReplicaCAD_dataset"

# Choose a scene to load (e.g., "apt_0" or "v3_sc2_staging_00")
scene_name = "apt_0"
scene_config_file = os.path.join(DATASET_PATH, "configs/scenes", f"{scene_name}.scene_instance.json")
with open(scene_config_file, 'r') as f:
    scene_data = json.load(f)

def parse_into_scene(scene):

    # Load the static stage (apartment architecture)
    stage_template = scene_data["stage_instance"]["template_name"]  # e.g. "stages/frl_apartment_stage" or "Stage_v3_sc2_staging"
    # Resolve stage asset path via its stage config
    stage_template_name = stage_template.split("/")[-1]  # e.g. "frl_apartment_stage"
    stage_config_path = os.path.join(DATASET_PATH, "configs/stages", f"{stage_template_name}.stage_config.json")
    with open(stage_config_path, 'r') as f:
        stage_config = json.load(f)
    stage_asset = stage_config["render_asset"]                      # e.g. "../../stages/frl_apartment_stage.glb"
    stage_asset_path = os.path.join(DATASET_PATH, "stages", os.path.basename(stage_asset))
    # Use friction/restitution from config if available, else defaults
    stage_friction = stage_config.get("friction_coefficient", 0.5)
    stage_restitution = stage_config.get("restitution_coefficient", 0.0)

    # Add stage as a fixed static mesh with collision
    stage_morph = gs.morphs.Mesh(
        file=stage_asset_path,
        pos=(0, 0, 0), euler=(90, 0, 0),       # rotate from Y-up to Z-up
        fixed=True, requires_jac_and_IK=False,
        collision=True, visualization=True,
        convexify=True, decompose_nonconvex=True, decimate=True, decimate_face_num=500,
        merge_submeshes_for_collision=True, group_by_material=True
    )
    scene.add_entity(
        morph=stage_morph,
        material=gs.materials.Rigid(friction=stage_friction, coup_restitution=stage_restitution),
        surface=gs.surfaces.Default(vis_mode="visual")
    )

    # Function to convert Habitat (Y-up) transform to Genesis (Z-up)
    def habitat_to_genesis_transform(translation, rotation_quat):
        """Convert position and quaternion from Habitat (Y-up) to Genesis (Z-up) frame."""
        # Rotation of 90 deg about X-axis
        Rx90 = np.array([[1, 0, 0],
                        [0, 0, -1],
                        [0, 1,  0]])
        # Position: apply Rx90
        trans = np.array(translation, dtype=float)
        new_pos = Rx90.dot(trans)  # (x, y, z)_gen = (x, -z, y)_hab
        # Quaternion: multiply with X-90 quaternion (w,x,y,z format)
        # 90° about X-axis quaternion:
        qx = np.array([0.70710678, 0.70710678, 0.0, 0.0])  # w=cos(45°), x=sin(45°), y=0, z=0
        q = np.array(rotation_quat, dtype=float)
        # Quaternion multiplication: qx * q  (both in w,x,y,z)
        w1,x1,y1,z1 = qx;  w2,x2,y2,z2 = q
        new_quat = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=float)
        # Normalize quaternion (to avoid numerical drift)
        new_quat = new_quat / np.linalg.norm(new_quat)
        return tuple(new_pos), tuple(new_quat)

    # Add all object instances (treat all as static obstacles for stability)
    for obj in scene_data.get("object_instances", []):
        template = obj["template_name"]            # e.g. "objects/frl_apartment_sofa"
        # Resolve object config and asset paths
        obj_name = template.split("/")[-1]         # e.g. "frl_apartment_sofa"
        obj_cfg_path = os.path.join(DATASET_PATH, "configs/objects", f"{obj_name}.object_config.json")
        with open(obj_cfg_path, 'r') as f:
            obj_cfg = json.load(f)
        visual_asset = os.path.join(DATASET_PATH, "objects", os.path.basename(obj_cfg["render_asset"]))
        # Get physical properties if provided
        friction = obj_cfg.get("friction_coefficient", 0.5)
        restitution = obj_cfg.get("restitution_coefficient", 0.0)
        # Habitat transform (quaternion is [w,x,y,z] in config)
        translation = obj.get("translation", [0, 0, 0])
        rotation = obj.get("rotation", [1, 0, 0, 0])
        pos_gen, quat_gen = habitat_to_genesis_transform(translation, rotation)
        # Add the object as a fixed rigid body with collision
        obj_morph = gs.morphs.Mesh(
            file=visual_asset,
            pos=pos_gen, quat=quat_gen,
            scale=obj_cfg.get("scale", 1.0),        # use scale if specified in config
            fixed=True, requires_jac_and_IK=False,
            collision=True, visualization=True,
            convexify=True, decompose_nonconvex=True, decimate=True, decimate_face_num=200,
            merge_submeshes_for_collision=True, group_by_material=True
        )
        scene.add_entity(
            morph=obj_morph,
            material=gs.materials.Rigid(friction=friction, coup_restitution=restitution),
            surface=gs.surfaces.Default(vis_mode="visual")
        )

    # Add articulated objects (with URDFs) such as doors, cabinets, etc.
    for i, art in enumerate(scene_data.get("articulated_object_instances", [])):
        print(f">>> {i} | art: {art}")
        name = art["template_name"]  # e.g. "fridge", "chestOfDrawers_01", "door2"
        # Determine URDF path based on name and dataset structure
        if name.startswith("chestOfDrawers"):
            urdf_path = os.path.join(DATASET_PATH, "urdf/chest_of_drawers", f"{name}.urdf")
        elif name.startswith("kitchenCupboard"):
            urdf_path = os.path.join(DATASET_PATH, "urdf/kitchen_cupboards", f"{name}.urdf")
        elif name.startswith("door"):  # door1, door2, etc.
            urdf_path = os.path.join(DATASET_PATH, "urdf/doors", f"{name}.urdf")
        else:
            # For names matching folder directly (fridge, cabinet, kitchen_counter, etc.)
            urdf_path = os.path.join(DATASET_PATH, f"urdf/{name}", f"{name}.urdf")
        if not os.path.exists(urdf_path):
            print(f"URDF not found for {name}: expected at {urdf_path}")
            continue
        # Get transform and convert to Genesis frame
        translation = art.get("translation", [0, 0, 0])
        rotation = art.get("rotation", [1, 0, 0, 0])
        pos_gen, quat_gen = habitat_to_genesis_transform(translation, rotation)
        # Add URDF-based articulated object (base fixed in place)
        art_morph = gs.morphs.URDF(file=urdf_path, pos=pos_gen, quat=quat_gen, fixed=True)
        scene.add_entity(
            morph=art_morph,
            material=gs.materials.Rigid(friction=0.5, coup_restitution=0.0),
            surface=gs.surfaces.Default(vis_mode="visual")
        )
