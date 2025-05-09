import bpy
import json
from mathutils import Vector
import os


current_file_path = bpy.data.filepath
current_file_dir = os.path.dirname(current_file_path)

part_dict_path = os.path.join(current_file_dir, "part_to_obj.json")

with open(part_dict_path, "r") as openfile:
    part_dict = json.load(openfile)

material_names = [a["material"] for a in part_dict.values()]

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

save_root_path = os.path.join(current_file_dir, "per_material")

if not os.path.exists(save_root_path):
    os.makedirs(save_root_path)

object_names = bpy.context.scene.objects.keys()

for material_name in material_names:
    bpy.ops.object.select_all(action="DESELECT")
    for j, obj_name in enumerate(object_names):
        obj = bpy.data.objects[obj_name]
        assert len(obj.material_slots) == 1, "Multiple material in one object"
        if material_name == obj.material_slots[0].name:
            obj.select_set(True)
    assert len(bpy.context.selected_objects) > 0, "Material not used by any object"

    save_path = os.path.join(save_root_path, material_name)
    bpy.ops.wm.collada_export(filepath=save_path, selected=True)
