import os, sys, datetime, json
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(gpus[0], "GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.keras.utils.set_random_seed(1115)
tf.config.experimental.enable_op_determinism()
tf.get_logger().setLevel("ERROR")
tf.config.run_functions_eagerly(False)

from sionna.channel import subcarrier_frequencies

from utils.utils import *

from wrapper.scene_wrapper import SceneWrapper
from trainable.neural_scene import NeuralScene
from trainable.utils import LRSchedule

from utils.training import training_step, evaluation_step

training_name = "office_v1"

# Dataset
dataset_name = "office_v1"
dataset_folder = "data/traced_paths"
params_filename = os.path.join(f"{dataset_folder}/{dataset_name}" + ".json")
dataset_filename = os.path.join(f"{dataset_folder}/{dataset_name}" + ".tfrecords")

# Training parameters
num_epoch = 200
batch_size = 100
learning_rate = 4e-4
training_set_size = 25600
validation_set_size = 1000
test_set_size = 1000
warm_up_end = 2000
end_iter = int(num_epoch * training_set_size // batch_size)


position_encoding_size = 10


with open(params_filename, "r") as openfile:
    params = json.load(openfile)

# Scene
scene_path = params["scene_path"]
part_to_obj_dict, obj_to_part_dict = load_obj_part_dicts(
    "/".join(scene_path.split("/")[:-1])
)
# Size of the dataset
dataset_size = params["traced_paths_dataset_size"]
dataset_filename = (
    os.path.join(params["traced_paths_dataset_folder"], params["traced_paths_dataset"])
    + ".tfrecords"
)

num_subcarriers = 1024
bandwidth = 50e6
frequencies = subcarrier_frequencies(num_subcarriers, bandwidth / num_subcarriers)

# Load dataset
dataset = tf.data.TFRecordDataset([dataset_filename]).map(
    deserialize_paths_as_tensor_dicts
)
shuffled_dataset = dataset.shuffle(256, seed=1115, reshuffle_each_iteration=False)
training_set = shuffled_dataset.take(training_set_size).cache()
validation_set = (
    shuffled_dataset.skip(training_set_size).take(validation_set_size).cache()
)
test_set = shuffled_dataset.skip(dataset_size - test_set_size).cache()

# Training set
training_set = (
    training_set.shuffle(25600, seed=1115)
    .batch(batch_size, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)
num_training_batch = training_set_size // batch_size

# Validation set
validation_set = validation_set.batch(batch_size, drop_remainder=True).prefetch(
    tf.data.AUTOTUNE
)
num_validation_batch = validation_set_size // batch_size

# Test set
test_set = test_set.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
num_test_batch = test_set_size // batch_size

# Load the scene
scene = SceneWrapper(scene_path)

# Print scene details
print(f"Total number of sionna objects: {len(scene.objects)}")

# Add Tx
scene.add(
    Transmitter(name=f"tx-{1}", position=params["tx_pos"], orientation=[0.0, 0.0, 0.0])
)

# Add Rx
for i in range(batch_size):
    name = f"rx-{i}"
    if scene.get(name) is None:
        scene.add(Receiver(name=f"rx-{i}", position=(0.0, 0.0, 0.0)))


# Optimizer
lr_schedule = LRSchedule(
    learning_rate=learning_rate,
    end_iter=end_iter,
    warm_up_end=warm_up_end,
    alpha=0.05,
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=5.0)


# Setting up tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join("tb_logs/", training_name, current_time)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# Checkpoint
weights_filename = os.path.join("checkpoints/", training_name)

# Model
scene.neural_scene = NeuralScene(
    scene,
    part_to_obj_dict=part_to_obj_dict,
    obj_to_part_dict=obj_to_part_dict,
    pos_encoding_size=position_encoding_size,
    learn_scattering=True,
)

scene.neural_scene(
    tf.zeros([1, 1, 1], tf.int32), tf.zeros([1, 1, 1, 3]), tf.zeros([1, 1, 1, 1]), 0
)
scene.neural_scene(
    tf.zeros([1, 1, 1], tf.int32), tf.zeros([1, 1, 1, 3]), tf.zeros([1, 1, 1, 7]), 1
)
scene.neural_scene(
    tf.zeros([1, 1, 1], tf.int32), tf.zeros([1, 1, 1, 3]), tf.zeros([1, 1, 1, 13]), 2
)


# Training loop
for epoch in range(num_epoch):  # loop over the dataset multiple times
    running_loss_nmse = 0.0
    running_loss_log_nmse = 0.0
    with tqdm(
        training_set, unit="batch", file=sys.stdout, total=num_training_batch
    ) as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for i, next_item in enumerate(tepoch, 0):
            rx_pos, h_meas, traced_paths = next_item[0], next_item[1], next_item[2:]
            # Skip iteration if does not match the batch size
            if rx_pos.shape[0] != batch_size:
                continue

            # Batchify
            traced_paths = batchify(traced_paths)

            # Train
            loss_nmse, loss_log_nmse = training_step(
                rx_pos,
                h_meas,
                traced_paths,
                scene,
                frequencies,
                optimizer,
                scattering=params["scattering"],
            )

            running_loss_nmse = (loss_nmse.numpy().item() + i * running_loss_nmse) / (
                i + 1
            )
            running_loss_log_nmse = (
                loss_log_nmse.numpy().item() + i * running_loss_log_nmse
            ) / (i + 1)

            log = OrderedDict()
            log["loss_nmse"] = running_loss_nmse
            tepoch.set_postfix(log)

        # Evaluate periodically on the evaluation set
        eval_loss_nmse = 0.0
        eval_all_nmse = []
        for next_item in validation_set:
            # Retreive the receiver position separately
            rx_pos, h_meas, traced_paths = next_item[0], next_item[1], next_item[2:]

            # Batchify
            traced_paths = batchify(traced_paths)

            # Test
            loss_nmse, all_nmse = evaluation_step(
                rx_pos,
                h_meas,
                traced_paths,
                scene,
                frequencies,
                scattering=params["scattering"],
            )
            eval_loss_nmse += loss_nmse
            eval_all_nmse.append(all_nmse)
        eval_loss_nmse /= float(num_validation_batch)
        eval_all_nmse_sort = tf.sort(tf.concat(eval_all_nmse, axis=0))
        eval_all_nmse_fil = eval_all_nmse_sort[: int(validation_set_size * 0.99)]
        eval_all_nmse_fil = tf.reduce_mean(eval_all_nmse_fil)

        eval_log_nmse = tf.reduce_mean(
            10 * tf.experimental.numpy.log10(eval_all_nmse_sort)
        )

        if ((epoch % 10) == 0) or (epoch + 10 > num_epoch):
            # Save model
            save_model(
                scene.neural_scene,
                f"{weights_filename}_{epoch}.ckpt",
            )

    # Log
    with train_summary_writer.as_default():
        tf.summary.scalar("loss_nmse/train", running_loss_nmse, step=epoch)
        tf.summary.scalar("loss_log_nmse/train", running_loss_log_nmse, step=epoch)

        tf.summary.scalar("loss_nmse/eval", eval_all_nmse_fil, step=epoch)
        tf.summary.scalar("loss_log_nmse/eval", eval_log_nmse, step=epoch)

    save_model(scene.neural_scene, f"{weights_filename}.ckpt")

# Test
scene.neural_scene(
    tf.zeros([1, 1, 1], tf.int32), tf.zeros([1, 1, 1, 3]), tf.zeros([1, 1, 1, 1]), 0
)
scene.neural_scene(
    tf.zeros([1, 1, 1], tf.int32), tf.zeros([1, 1, 1, 3]), tf.zeros([1, 1, 1, 7]), 1
)
scene.neural_scene(
    tf.zeros([1, 1, 1], tf.int32), tf.zeros([1, 1, 1, 3]), tf.zeros([1, 1, 1, 13]), 2
)

load_model(scene.neural_scene, f"{weights_filename}.ckpt")
# load_model(scene.neural_scene, "checkpoints/office_v1_95.ckpt")

eval_loss_nmse = 0.0
eval_all_nmse = []
for i, next_item in enumerate(test_set):
    # Retreive the receiver position separately
    rx_pos, h_meas, traced_paths = next_item[0], next_item[1], next_item[2:]

    # Batchify
    traced_paths = batchify(traced_paths)

    # Test
    loss_nmse, loss_nmse_all = evaluation_step(
        rx_pos,
        h_meas,
        traced_paths,
        scene,
        frequencies,
        scattering=params["scattering"],
    )
    eval_all_nmse.append(loss_nmse_all)
    eval_loss_nmse += loss_nmse

eval_all_nmse = tf.concat(eval_all_nmse, axis=0).cpu().numpy()
eval_all_nmse_db = 10 * np.log10(eval_all_nmse)
cdf = np.arange(1, len(eval_all_nmse_db) + 1) / len(eval_all_nmse_db)

plt.plot(np.sort(eval_all_nmse_db), cdf)
plt.xlabel("Channel NMSE (dB)")
plt.ylabel("Emperical CDF")
plt.grid(True)
plt.savefig("cdf.png")
