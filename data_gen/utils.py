import tensorflow as tf
from sionna.rt import load_scene, PlanarArray


def serialize_traced_paths(rx_pos, h_meas, traced_paths, squeeze_target_dim):
    # Map Paths objects to a single dictionary of tensor
    dict_list_ = [x.to_dict() for x in traced_paths]

    # Target axis index
    target_axis = {
        "a": 0,
        "tau": 0,
        "mask": 0,
        "objects": 1,
        "phi_r": 0,
        "phi_t": 0,
        "theta_r": 0,
        "theta_t": 0,
        "vertices": 1,
        "targets": 0,
        # Paths TMP data
        "normals": 1,
        "k_i": 1,
        "k_r": 1,
        "total_distance": 0,
        "mat_t": 0,
        "k_tx": 0,
        "k_rx": 0,
        "scat_last_objects": 0,
        "scat_last_vertices": 0,
        "scat_last_k_i": 0,
        "scat_k_s": 0,
        "scat_last_normals": 0,
        "scat_src_2_last_int_dist": 0,
        "scat_2_target_dist": 0,
    }

    # Remove useless tensors and drop the target axis
    dict_list = []
    for d_ in dict_list_:
        d = {}
        for k in d_.keys():
            # Drop useless tensors
            if not k.startswith("scat_prefix_"):
                d.update({k: d_[k]})
                # Squeezes target dimension if requested
                if squeeze_target_dim and (k in target_axis):
                    d[k] = tf.squeeze(d[k], axis=target_axis[k])
        dict_list.append(d)

    # Add a prefix to indicate to which object each tensor belongs to
    all_tensors = {}
    all_tensors.update({"spec-" + k: dict_list[0][k] for k in dict_list[0]})
    all_tensors.update({"diff-" + k: dict_list[1][k] for k in dict_list[1]})
    all_tensors.update({"scat-" + k: dict_list[2][k] for k in dict_list[2]})
    all_tensors.update({"tmp-spec-" + k: dict_list[3][k] for k in dict_list[3]})
    all_tensors.update({"tmp-diff-" + k: dict_list[4][k] for k in dict_list[4]})
    all_tensors.update({"tmp-scat-" + k: dict_list[5][k] for k in dict_list[5]})

    # Add the receiver position
    all_tensors.update({"rx_pos": rx_pos})

    # Add the channel measurement
    all_tensors.update({"h_meas": h_meas})

    # Serialize the tensors to a string of bytes
    for k, v in all_tensors.items():
        all_tensors[k] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(v).numpy()])
        )

    ex = tf.train.Example(features=tf.train.Features(feature=all_tensors))
    record_bytes = ex.SerializeToString()
    return record_bytes


def pad_traced_paths(
    traced_paths, max_num_paths_spec, max_num_paths_diff, max_num_paths_scat, max_depth
):

    ### Function that pads a `Paths` object
    def _pad_paths(paths, max_num_paths, max_depth):
        axis_to_pad = {
            "mask": {"axis": [-1], "padded_size": [max_num_paths]},
            "objects": {"axis": [-3, -1], "padded_size": [max_depth, max_num_paths]},
            "phi_r": {"axis": [-1], "padded_size": [max_num_paths]},
            "phi_t": {"axis": [-1], "padded_size": [max_num_paths]},
            "tau": {"axis": [-1], "padded_size": [max_num_paths]},
            "theta_r": {"axis": [-1], "padded_size": [max_num_paths]},
            "theta_t": {"axis": [-1], "padded_size": [max_num_paths]},
            "vertices": {"axis": [-4, -2], "padded_size": [max_depth, max_num_paths]},
        }

        paths_dicts = paths.to_dict()
        for k, a in axis_to_pad.items():
            t = paths_dicts[k]

            r = tf.rank(t).numpy()
            dims = r + a["axis"]

            # if num_paths == 0:
            #     continue

            padding_size = [
                a["padded_size"][i] - t.shape[d] for i, d in enumerate(dims)
            ]
            assert min(padding_size) >= 0

            paddings = [[0, 0]] * r
            for d, p in zip(dims, padding_size):
                paddings[d] = [0, p]

            if k == "mask":
                t = tf.pad(
                    t, paddings, constant_values=False
                )  # Mask the paths added for padding
            elif k == "tau":
                t = tf.pad(t, paddings, constant_values=-1.0)
            elif k == "objects":
                t = tf.pad(t, paddings, constant_values=-1)
            else:
                t = tf.pad(t, paddings, constant_values=0)
            paths_dicts[k] = t
        paths.from_dict(paths_dicts)

        return paths

    ### Function that pads a `PathsTmpData` object
    def _pad_tmp_paths(tmp_paths, max_num_paths, max_depth):
        axis_to_pad = {
            "k_i": {"axis": [-4, -2], "padded_size": [max_depth + 1, max_num_paths]},
            "k_r": {"axis": [-4, -2], "padded_size": [max_depth, max_num_paths]},
            "k_rx": {"axis": [-2], "padded_size": [max_num_paths]},
            "k_tx": {"axis": [-2], "padded_size": [max_num_paths]},
            "normals": {"axis": [-4, -2], "padded_size": [max_depth, max_num_paths]},
            "scat_2_target_dist": {"axis": [-1], "padded_size": [max_num_paths]},
            "scat_k_s": {"axis": [-2], "padded_size": [max_num_paths]},
            "scat_last_k_i": {"axis": [-2], "padded_size": [max_num_paths]},
            "scat_last_normals": {"axis": [-2], "padded_size": [max_num_paths]},
            "scat_last_objects": {"axis": [-1], "padded_size": [max_num_paths]},
            "scat_last_vertices": {"axis": [-2], "padded_size": [max_num_paths]},
            "scat_src_2_last_int_dist": {"axis": [-1], "padded_size": [max_num_paths]},
            "total_distance": {"axis": [-1], "padded_size": [max_num_paths]},
        }

        paths_dicts = tmp_paths.to_dict()
        for k, a in axis_to_pad.items():
            t = paths_dicts[k]

            r = tf.rank(t).numpy()
            dims = r + a["axis"]

            # if num_paths == 0:
            #     continue

            padding_size = [
                a["padded_size"][i] - t.shape[d] for i, d in enumerate(dims)
            ]
            assert min(padding_size) >= 0

            paddings = [[0, 0]] * r
            for d, p in zip(dims, padding_size):
                paddings[d] = [0, p]

            t = tf.pad(t, paddings, constant_values=0)
            paths_dicts[k] = t
        tmp_paths.from_dict(paths_dicts)

        return tmp_paths

    # Tiling the paths
    spec_paths = _pad_paths(traced_paths[0], max_num_paths_spec, max_depth)
    diff_paths = _pad_paths(traced_paths[1], max_num_paths_diff, 1)
    scat_paths = _pad_paths(traced_paths[2], max_num_paths_scat, max_depth)

    # Tiling the additional data paths
    tmp_spec_paths = _pad_tmp_paths(traced_paths[3], max_num_paths_spec, max_depth)
    tmp_diff_paths = _pad_tmp_paths(traced_paths[4], max_num_paths_diff, 1)
    tmp_scat_paths = _pad_tmp_paths(traced_paths[5], max_num_paths_scat, max_depth)

    return (
        spec_paths,
        diff_paths,
        scat_paths,
        tmp_spec_paths,
        tmp_diff_paths,
        tmp_scat_paths,
    )


def init_scene(path, tx_pattern="tr38901", rx_pattern="dipole"):
    scene = load_scene(path)
    scene.frequency = 3.438e9
    scene.synthetic_array = False

    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=tx_pattern,
        polarization="V",
    )

    # This is the antenna used by the measurement robot
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=rx_pattern,
        polarization="V",
    )
    return scene


def get_num_paths(traced_paths):
    # the traced_paths should be the result of a single RX
    num_los_specular = 0
    num_diffraction = 0
    num_scattering = 0

    if traced_paths[0].mask.shape[-1]:
        num_los_specular = traced_paths[0].mask.numpy().sum()

    if traced_paths[1].mask.shape[-1]:
        num_diffraction = traced_paths[1].mask.numpy().sum()

    if traced_paths[2].mask.shape[-1]:
        num_scattering = traced_paths[2].mask.numpy().sum()

    total_num_path = num_los_specular + num_diffraction + num_scattering

    return total_num_path
