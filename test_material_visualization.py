import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(gpus[0], "GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.keras.utils.set_random_seed(1115)
tf.config.experimental.enable_op_determinism()
tf.get_logger().setLevel("ERROR")
tf.config.run_functions_eagerly(False)

from utils.utils import *
from wrapper.scene_wrapper import SceneWrapper
from trainable.neural_scene import NeuralScene


def plot_figs(object_points, labels_mat, num_mat, fig_save_path=False):
    # plot colored 3D points
    xs = object_points[:, 0].numpy()
    ys = object_points[:, 1].numpy()
    zs = object_points[:, 2].numpy()
    cmap = plt.get_cmap("viridis", num_mat)

    # plot colored 3D points
    fig1 = plt.figure(figsize=(20, 18))
    ax1 = fig1.add_subplot(projection="3d")
    p1 = ax1.scatter(xs, ys, zs, c=labels_mat, cmap=cmap)
    ax1.set_title("3D plot", fontsize=20)
    ax1.set_xlabel("x-axis (m)", fontsize=20)
    ax1.set_ylabel("y-axis (m)", fontsize=20)
    ax1.set_zlabel("z-axis (m)", fontsize=20)
    ax1.set_title(object_name, fontsize=24)
    ax1.view_init(elev=50.0, azim=-150)
    ax1.axis("equal")

    # add colorbar
    cbar1 = plt.colorbar(p1)
    tick_locs = (np.arange(num_mat) + 0.5) * (num_mat - 1) / num_mat
    cbar1.set_ticks(tick_locs)
    cbar1.set_ticklabels(np.arange(num_mat), fontsize=20)
    if fig_save_path:
        fig1.savefig(f"{fig_save_path}/mat_rep.png")
    plt.close("all")


if __name__ == "__main__":
    training_name = "office_v1"

    # Dataset
    dataset_name = "office_v1"
    dataset_folder = "data/traced_paths"
    params_filename = os.path.join(f"{dataset_folder}/{dataset_name}" + ".json")
    dataset_filename = os.path.join(f"{dataset_folder}/{dataset_name}" + ".tfrecords")
    with open(params_filename, "r") as openfile:
        params = json.load(openfile)

    # Scene
    scene_path = params["scene_path"]
    part_to_obj_dict, obj_to_part_dict = load_obj_part_dicts(
        "/".join(scene_path.split("/")[:-1])
    )
    # Load the scene
    scene = SceneWrapper(scene_path)

    scene.neural_scene = NeuralScene(
        scene,
        part_to_obj_dict=part_to_obj_dict,
        obj_to_part_dict=obj_to_part_dict,
        pos_encoding_size=10,
        learn_scattering=True,
        verbose=True,
    )
    # Dummy run to build the model
    scene.neural_scene(
        tf.zeros([1, 1, 1], tf.int32), tf.zeros([1, 1, 1, 3]), tf.zeros([1, 1, 1, 1]), 0
    )
    scene.neural_scene(
        tf.zeros([1, 1, 1], tf.int32), tf.zeros([1, 1, 1, 3]), tf.zeros([1, 1, 1, 7]), 1
    )
    scene.neural_scene(
        tf.zeros([1, 1, 1], tf.int32),
        tf.zeros([1, 1, 1, 3]),
        tf.zeros([1, 1, 1, 13]),
        2,
    )
    load_model(scene.neural_scene, f"checkpoints/{training_name}.ckpt")

    for part_name in scene.objects.keys():
        part_to_obj_dict[part_name]["sionna_object_idx"] = scene.objects[
            part_name
        ].object_id

    for object_name in obj_to_part_dict:
        # if not object_name=="room": continue
        object_idx = obj_to_part_dict[object_name]["object_idx"]
        one_part_name = obj_to_part_dict[object_name]["parts"][0]
        sionna_object_idx = part_to_obj_dict[one_part_name]["sionna_object_idx"]

        all_material = []
        parts = obj_to_part_dict[object_name]["parts"]
        for part in parts:
            all_material.append(part_to_obj_dict[part]["material"])
        all_material_unique = list(set(all_material))
        num_mat = len(all_material_unique)
        all_material_unique = {all_material_unique[i]: i for i in range(num_mat)}

        all_point_material_idx = []
        object_points = []
        for part in parts:
            point_cloud_path = f"scenes/{dataset_name}/per_part/{part}.xyz"
            part_points = np.loadtxt(point_cloud_path)
            part_points = tf.constant(part_points, dtype=tf.float32)
            material_name = part_to_obj_dict[part]["material"]
            material_idx = all_material_unique[material_name]
            part_material_idx = (
                np.zeros((part_points.shape[0],), dtype=np.int32) + material_idx
            )
            all_point_material_idx.append(part_material_idx)
            object_points.append(part_points)
        all_point_material_idx = np.concatenate(all_point_material_idx)
        object_points = np.concatenate(object_points, 0)
        object_points = tf.constant(object_points, dtype=tf.float32)

        fig_save_path = (
            f"results/{training_name}/{dataset_name}/per_object/{object_name}"
        )
        os.makedirs(fig_save_path, exist_ok=True)
        with open(f"{fig_save_path}/material_name.json", "w") as outfile:
            json.dump(all_material_unique, outfile)

        sionna_object_idx = (
            tf.zeros(
                [
                    object_points.shape[0],
                ],
                tf.int32,
            )
            + sionna_object_idx
        )
        geo_feature = tf.zeros([object_points.shape[0], 1])
        _, neural_mat_rep_all = scene.neural_scene(
            sionna_object_idx, object_points, geo_feature, 0
        )

        neural_mat_rep_all = neural_mat_rep_all.numpy()

        est = KMeans(n_clusters=num_mat)
        est.fit(neural_mat_rep_all)
        labels_mat = est.labels_

        plot_figs(object_points, labels_mat, num_mat, fig_save_path=fig_save_path)

    print("done")
