import bpy
import json
from mathutils import Vector
import os


current_file_path = bpy.data.filepath
current_file_dir = os.path.dirname(current_file_path)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

save_root_path = os.path.join(current_file_dir, "per_object")

if not os.path.exists(save_root_path):
    os.makedirs(save_root_path)

collection_names = bpy.context.scene.collection.children.keys()
num_collections = len(collection_names)
print(f"Number of objects: {num_collections}")
print(collection_names)


for j, collection_name in enumerate(collection_names):
    bpy.ops.object.select_all(action="DESELECT")
    print(f"Collection {j}: {collection_name}")
    obj_names = bpy.context.scene.collection.children[collection_name].objects.keys()
    assert len(obj_names) > 0, "Empty collection!"

    for i, obj_name in enumerate(obj_names):

        print(f"Object {i}: {obj_name}")
        obj = bpy.data.objects[obj_name]
        obj.select_set(True)
    save_path = os.path.join(save_root_path, collection_name)
    bpy.ops.wm.collada_export(filepath=save_path, selected=True)
