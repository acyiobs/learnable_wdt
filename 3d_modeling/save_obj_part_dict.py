import bpy
import json
import os
from mathutils import Vector

current_file_path = bpy.data.filepath
current_file_dir = os.path.dirname(current_file_path)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

save_path = os.path.join(current_file_dir)

object_dict = {}
part_dict = {}

collection_names = bpy.context.scene.collection.children.keys()
num_collections = len(collection_names)
print(f"Number of objects: {num_collections}")
print(collection_names)


for j, collection_name in enumerate(collection_names):
    print(f"Collection {j}: {collection_name}")
    obj_names = bpy.context.scene.collection.children[collection_name].objects.keys()
    if not len(obj_names) > 0:
        break

    object_dict[collection_name.replace(".", "_")] = {"parts": []}

    obj_names = bpy.context.scene.collection.children[collection_name].objects.keys()

    box_max = [-float("inf"), -float("inf"), -float("inf")]
    box_min = [float("inf"), float("inf"), float("inf")]
    for i, obj_name in enumerate(obj_names):

        print(f"Object {i}: {obj_name}")
        obj = bpy.data.objects[obj_name]

        if i == 0:
            location = Vector(obj.location)
            rotation = Vector(obj.rotation_euler)
        else:
            assert location == Vector(obj.location), "Location does not match"
            assert rotation == Vector(obj.rotation_euler), "Rotation does not match"

        bbox_corners = [Vector(corner) for corner in obj.bound_box]

        box_min_ = []
        box_max_ = []
        for i in range(3):
            box_min_.append(min([b[i] for b in bbox_corners]))
            box_max_.append(max([b[i] for b in bbox_corners]))

        box_min = [min(box_min_[i], box_min[i]) for i in range(3)]
        box_max = [max(box_max_[i], box_max[i]) for i in range(3)]

        assert len(obj.material_slots) == 1
        material_name = obj.material_slots[0].name

        part_dict[obj_name.replace(".", "_")] = {
            "object": collection_name.replace(".", "_"),
            "material": material_name,
            "object_idx": j,
        }
        object_dict[collection_name.replace(".", "_")]["parts"].append(
            obj_name.replace(".", "_")
        )
        object_dict[collection_name.replace(".", "_")]["object_idx"] = j
        object_dict[collection_name.replace(".", "_")]["location"] = location.to_tuple()
        object_dict[collection_name.replace(".", "_")]["rotation"] = rotation.to_tuple()
        object_dict[collection_name.replace(".", "_")]["dimension"] = [box_min, box_max]

with open(f"{save_path}/obj_to_part.json", "w") as fp:
    object_dict_txt = json.dumps(object_dict)
    fp.write(object_dict_txt)

with open(f"{save_path}/part_to_obj.json", "w") as fp:
    part_dict_txt = json.dumps(part_dict)
    fp.write(part_dict_txt)
