import bpy
from mathutils import Vector
import numpy as np

obj_names = bpy.data.objects.keys()

for obj_name in obj_names:
    print(f"\nObject name: {obj_name}")
    obj = bpy.data.objects[obj_name]
    print(f"Location: {obj.location}")

    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    print(f'Bounding box": {bbox_corners}')

    box_dimensions = []
    for i in range(3):
        box_min = min([b[i] for b in bbox_corners])
        box_max = max([b[i] for b in bbox_corners])
        box_dimensions.append(box_max - box_min)
    box_dimensions = Vector(box_dimensions)
    print(f"Box dimensions: {box_dimensions}")

    local_bbox_center = 0.125 * sum((Vector(b) for b in obj.bound_box), Vector())
    global_bbox_center = obj.matrix_world @ local_bbox_center

    print(f"Global center: {global_bbox_center}")

    global_bbox_ground = global_bbox_center
    global_bbox_ground[2] -= box_dimensions[2] / 2
    print(f"Global ground: {global_bbox_ground}")

    print(f"Current Location: {obj.location}")

    if np.sum(np.array(global_bbox_ground) > 1e-5) > 0:
        print("!!!!!!!!! Change !!!!!!!!!!")
    obj.location -= global_bbox_ground

    print(f"New Location: {obj.location}")
    global_bbox_center = obj.matrix_world @ local_bbox_center
    print(f"New Global center: {global_bbox_center}")
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    # print(f'New Bounding box": {bbox_corners}')

    bpy.context.scene.cursor.location = Vector((0.0, 0.0, 0.0))
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    print(f"Box dimensions: {box_dimensions}")
