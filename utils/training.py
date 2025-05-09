import tensorflow as tf
import sionna
from sionna.channel import cir_to_ofdm_channel
from utils.utils import set_receiver_pos, tensor_dicts_to_traced_paths


@tf.function
def training_step(
    rx_pos, h_meas, traced_paths, scene, frequencies, optimizer, scattering=False
):
    # Placer receiver
    set_receiver_pos(scene, rx_pos)

    # Build traced paths
    traced_paths = tensor_dicts_to_traced_paths(scene, traced_paths)

    with tf.GradientTape() as tape:
        # Compute paths fields
        paths = scene.compute_fields(*traced_paths, check_scene=False)

        a, tau = paths.cir(scattering=scattering)  # Enable scattering

        # Compute channel frequency response
        h_rt = cir_to_ofdm_channel(frequencies, a, tau)

        # Remove useless dimensions
        h_rt = tf.squeeze(h_rt, axis=[0, 2, 5])

        # Normalize h to make sure that power is independent of the number of subacrriers
        h_rt /= tf.complex(tf.sqrt(tf.cast(tf.size(frequencies), tf.float32)), 0.0)

        # Compute losses
        h_rt = sionna.utils.flatten_dims(h_rt, 3, 0)
        h_meas = sionna.utils.flatten_dims(h_meas, 3, 0)

        loss_nmse = tf.reduce_sum(tf.abs(h_rt - h_meas) ** 2, axis=-1) / tf.reduce_sum(
            (tf.abs(h_meas) ** 2), axis=-1
        )
        loss_nmse = tf.reduce_mean(loss_nmse)
        loss_log_nmse = tf.reduce_mean(10 * tf.experimental.numpy.log10(loss_nmse))

    # Use loss_ds_pow for training
    grads = tape.gradient(
        loss_log_nmse,
        tape.watched_variables(),
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    return loss_nmse, loss_log_nmse


@tf.function
def evaluation_step(rx_pos, h_meas, traced_paths, scene, frequencies, scattering=True):
    # Placer receiver
    set_receiver_pos(scene, rx_pos)

    # Build traced paths
    traced_paths = tensor_dicts_to_traced_paths(scene, traced_paths)

    paths = scene.compute_fields(*traced_paths, check_scene=False)

    a, tau = paths.cir(scattering=scattering)  # Disable scattering

    # Compute channel frequency response
    h_rt = cir_to_ofdm_channel(frequencies, a, tau)

    # Remove useless dimensions
    h_rt = tf.squeeze(h_rt, axis=[0, 2, 5])

    # Normalize h to make sure that power is independent of the number of subacrriers
    h_rt /= tf.complex(tf.sqrt(tf.cast(tf.size(frequencies), tf.float32)), 0.0)

    # Compute losses
    h_rt = sionna.utils.flatten_dims(h_rt, 3, 0)
    h_meas = sionna.utils.flatten_dims(h_meas, 3, 0)

    loss_nmse = tf.reduce_sum(tf.abs(h_rt - h_meas) ** 2, axis=-1) / tf.reduce_sum(
        (tf.abs(h_meas) ** 2), axis=-1
    )
    loss_nmse_all = loss_nmse
    loss_nmse = tf.reduce_mean(loss_nmse)

    return loss_nmse, loss_nmse_all
