import os
import pymeshlab

dataset_root = "../scenes/office_v1"
radius = pymeshlab.PercentageValue(0.5)

dae_files = []
for root, dirs, files in os.walk(dataset_root):
    for file in files:
        if file.endswith(".dae"):
            dae_files.append(os.path.join(root, file))

for datafile in dae_files:
    print(datafile)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(datafile)
    ms.generate_sampling_poisson_disk(radius=radius)
    ms.set_current_mesh(1)
    ms.save_current_mesh(datafile.replace("dae", "xyz"), save_vertex_normal=False)
    ms.clear
