# read params
import os, json
from tqdm import tqdm
import numpy as np
import pandas as pd

from config_params import config_params

params = config_params()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

tf.random.set_seed(params["seed"])
gpus = tf.config.list_physical_devices("GPU")
print(gpus)
tf.config.set_visible_devices(gpus[0], "GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.get_logger().setLevel("ERROR")
tf.config.run_functions_eagerly(False)

from sionna.rt import BackscatteringPattern
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel

import numpy as np
from utils.utils import *
from .utils import init_scene, serialize_traced_paths, pad_traced_paths, get_num_paths

###########################################
############# Setup the scene #############
###########################################

# Load the scene
scene = init_scene(params["scene_path"])

scattering_pattern = BackscatteringPattern(
    alpha_r=params["scattering_pattern"]["alpha_r"],
    alpha_i=params["scattering_pattern"]["alpha_i"],
    lambda_=params["scattering_pattern"]["lambda_"],
)

# Print scene details
for obj in scene.objects.values():
    scene.objects[obj.name].radio_material.scattering_coefficient = params[
        "scattering_coefficient"
    ]
    scene.objects[obj.name].radio_material.xpd_coefficient = params["xpd_coefficient"]
    scene.objects[obj.name].radio_material.scattering_pattern = scattering_pattern
print(f"Total number of objects: {len(scene.objects)}")

# Add Tx
scene.add(
    Transmitter(name=f"tx-{0}", position=params["tx_pos"], orientation=[0.0, 0.0, 0.0])
)
# Add Rx
scene.add(Receiver(name=f"rx-{0}", position=(0.0, 0.0, 0.0)))

# Load rx positions
rx_pos_all = pd.read_csv(params["rx_pos_path"]).values.astype(np.float32)

###########################################
########## Genearate the dataset ##########
###########################################

num_subcarriers = 1024
bandwidth = 50e6
frequencies = subcarrier_frequencies(num_subcarriers, bandwidth / num_subcarriers)

traced_paths_raw_dataset_datafile = os.path.join(
    params["traced_paths_dataset_folder"],
    params["traced_paths_dataset"] + "_raw.tfrecords",
)

# File writer to save the dataset
file_writer = tf.io.TFRecordWriter(traced_paths_raw_dataset_datafile)

# Keep track of the max_num_paths
max_num_paths_spec = -1
max_num_paths_diff = -1
max_num_paths_scat = -1

num_valid_data = 0
with tqdm(total=params["traced_paths_dataset_size"]) as progress_bar:
    for i in range(rx_pos_all.shape[0]):
        if num_valid_data >= params["traced_paths_dataset_size"]:
            break

        rx_pos = rx_pos_all[i, :]
        # Place the receiver
        set_receiver_pos(scene, np.expand_dims(rx_pos, 0))

        # Trace the paths
        traced_paths = scene.trace_paths(
            num_samples=params["num_samples"],
            max_depth=params["max_depth"],
            los=params["los"],
            reflection=params["reflection"],
            diffraction=params["diffraction"],
            edge_diffraction=params["edge_diffraction"],
            scattering=params["scattering"],
            scat_keep_prob=params["scat_keep_prob"],
            check_scene=False,
        )

        # Drop the rx position that does not have any path with the tx
        if get_num_paths(traced_paths) == 0:
            continue

        # Compute field
        paths = scene.compute_fields(*traced_paths, scat_random_phases=False)
        path_mask = paths.mask.numpy().squeeze()  # false for non-existent path
        rx_num_path = np.sum(path_mask)

        # Compute channel impulse response
        a, tau = paths.cir()

        # Compute channel frequency response
        h_gt = cir_to_ofdm_channel(frequencies, a, tau)
        h_gt = tf.squeeze(h_gt, axis=[0, 1, 2, 5])
        # Normalize h to make sure that power is independent of the number of subacrriers
        h_gt /= tf.complex(tf.sqrt(tf.cast(num_subcarriers, tf.float32)), 0.0)

        # Update max_num_paths
        num_paths_spec = traced_paths[0].objects.shape[-1]
        num_paths_diff = traced_paths[1].objects.shape[-1]
        num_paths_scat = traced_paths[2].objects.shape[-1]

        if num_paths_spec > max_num_paths_spec:
            max_num_paths_spec = num_paths_spec
        if num_paths_diff > max_num_paths_diff:
            max_num_paths_diff = num_paths_diff
        if num_paths_scat > max_num_paths_scat:
            max_num_paths_scat = num_paths_scat

        # Serialize the traced paths
        record_bytes = serialize_traced_paths(rx_pos, h_gt, traced_paths, True)

        # Save the traced paths
        file_writer.write(record_bytes)

        # Print progress
        num_valid_data += 1
        progress_bar.update(1)


file_writer.close()
print("")
print(f"Raw dataset generated. Total number of data points: {num_valid_data}")
print(
    f"Maximum number of paths:\n\tLoS + Specular: {max_num_paths_spec}\n\tDiffracted: {max_num_paths_diff}\n\tScattered: {max_num_paths_scat}"
)


############################################
## Post-process the generated raw dataset ##
############################################

print("Post-processing the raw dataset...")

raw_dataset = tf.data.TFRecordDataset([traced_paths_raw_dataset_datafile])
raw_dataset = raw_dataset.map(deserialize_paths_as_tensor_dicts)


# Iterate through all the dataset and tile the paths to the same max_num_paths
# File writer to save the dataset
traced_paths_dataset_datafile = os.path.join(
    params["traced_paths_dataset_folder"], params["traced_paths_dataset"] + ".tfrecords"
)
file_writer = tf.io.TFRecordWriter(traced_paths_dataset_datafile)
for i, data in tqdm(enumerate(iter(raw_dataset)), total=num_valid_data):
    # Retreive the receiver position separately
    rx_pos, h_gt, traced_paths = data[0], data[1], data[2:]

    # Build traced paths
    traced_paths = tensor_dicts_to_traced_paths(scene, traced_paths)

    traced_paths = pad_traced_paths(
        traced_paths,
        max_num_paths_spec,
        max_num_paths_diff,
        max_num_paths_scat,
        max_depth=3,
    )
    record_bytes = serialize_traced_paths(rx_pos, h_gt, traced_paths, False)

    # Save the tiled traced paths
    file_writer.write(record_bytes)
file_writer.close()
print("")


#########################################
###### Save the dataset properties ######
#########################################

# Filename for storing the dataset parameters
params_filename = os.path.join(
    params["traced_paths_dataset_folder"], params["traced_paths_dataset"] + ".json"
)

# Add the maximum number of paths
params["max_num_paths_spec"] = max_num_paths_spec
params["max_num_paths_diff"] = max_num_paths_diff
params["max_num_paths_scat"] = max_num_paths_scat

# Dump the dataset parameters in a JSON file
with open(params_filename, "w") as outfile:
    json.dump(params, outfile)
