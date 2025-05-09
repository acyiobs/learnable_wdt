import pickle, os, json
import numpy as np
import tensorflow as tf
from sionna.rt import Transmitter, Receiver, Paths
from sionna.rt.solver_paths import PathsTmpData

def save_model(layer, filename):
    weights = layer.get_weights()
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)

def load_model(layer, filename):
    with open(filename, 'rb') as f:
        weights = pickle.load(f)
    layer.set_weights(weights)

def set_receiver_pos(scene, rx_pos):
    num_rx = rx_pos.shape[0]
    for i in range(num_rx):
        name = f'rx-{i}'
        scene.receivers[name].position = rx_pos[i]

def batchify(traced_paths_dicts):
    """
    Batchifies traced paths dictionaries

    This utility enables sampling batches of receivers positions from the dataset of traced paths.
    It arranges the receivers as targets, by concatenating and reshaping the tensors accordingly.

    Input
    ------
    tensor_dicts : `list` of `dict`
        List of dictionaries, as retrieved when iterating over a dataset
        of traced paths and using ``deserialize_paths_as_tensor_dicts()``
        to retrieve the data.

    Output
    -------
     : `list` of `dict`
        List of dictionaries
    """

    # Target axis index
    axis_to_swap = {'objects' : 1,
                    'vertices' : 1,
                    'k_i' : 1,
                    'k_r' : 1,
                    'normals' : 1}

    for d in traced_paths_dicts:
        # Swap axis 0 and targets if required
        # This is done because when batching a TF dataset, the batch dimension
        # is always axis 0, which might not be the target axis.
        for k in d.keys():
            if k not in axis_to_swap:
                continue
            v = d[k]
            a = axis_to_swap[k]
            perm = tf.range(tf.rank(v))
            perm = tf.tensor_scatter_nd_update(perm, [[0]], [a])
            perm = tf.tensor_scatter_nd_update(perm, [[a]], [0])
            d[k] = tf.transpose(v, perm)

        # Drop the batch dimension for sources, as these are the same
        # for all examples in the batch
        if 'sources' in d:
            d['sources'] = d['sources'][0]

        # Drop the batch dim for types
        if 'types' in d:
            d['types'] = d['types'][0]

    # De-batchify num_samples and scat_keep_prop
    traced_paths_dicts[-1]['num_samples'] = traced_paths_dicts[-1]['num_samples'][0]
    traced_paths_dicts[-1]['scat_keep_prob'] = traced_paths_dicts[-1]['scat_keep_prob'][0]
    traced_paths_dicts[-2]['num_samples'] = traced_paths_dicts[-2]['num_samples'][0]
    traced_paths_dicts[-2]['scat_keep_prob'] = traced_paths_dicts[-2]['scat_keep_prob'][0]
    traced_paths_dicts[-3]['num_samples'] = traced_paths_dicts[-3]['num_samples'][0]
    traced_paths_dicts[-3]['scat_keep_prob'] = traced_paths_dicts[-3]['scat_keep_prob'][0]

    return traced_paths_dicts

def deserialize_paths_as_tensor_dicts(serialized_item):
    """
    Deserializes examples of a dataset of traced paths

    Input
    -----
    serialized_item : str
        A stream of bytes

    Output
    -------
    rx_pos : [3], tf.float
        Position of the receiver

    h_meas : [num_tx*num_tx_ant=64, num_subcarriers=1024], tf.complex
        Measured CSI

    spec_data : dict
        Dictionary of LoS and specular traced paths

    diff_data : dict
        Dictionary of diffracted traced paths

    scat_data : dict
        Dictionary of scattered traced paths

    tmp_spec_data : dict
        Dictionary of additional LoS and specular traced paths data

    tmp_diff_data : dict
        Dictionary of additional diffracted traced paths data

    tmp_scat_data : dict
        Dictionary of additional scattered traced paths data
    """

    # Fields names and types
    paths_fields_dtypes = {'a' : tf.complex64,
                           'mask' : tf.bool,
                           'normalize_delays' : tf.bool,
                           'objects' : tf.int32,
                           'phi_r' : tf.float32,
                           'phi_t' : tf.float32,
                           'reverse_direction' : tf.bool,
                           'sources' : tf.float32,
                           'targets' : tf.float32,
                           'tau' : tf.float32,
                           'theta_r' : tf.float32,
                           'theta_t' : tf.float32,
                           'types' : tf.int32,
                           'vertices' : tf.float32}
    tmp_paths_fields_dtypes = {'k_i' : tf.float32,
                               'k_r' : tf.float32,
                               'k_rx' : tf.float32,
                               'k_tx' : tf.float32,
                               'normals' : tf.float32,
                               'scat_2_target_dist' : tf.float32,
                               'scat_k_s' : tf.float32,
                               'scat_last_k_i' : tf.float32,
                               'scat_last_normals' : tf.float32,
                               'scat_last_objects' : tf.int32,
                               'scat_last_vertices' : tf.float32,
                               'scat_src_2_last_int_dist' : tf.float32,
                               'sources' :tf.float32,
                               'targets' : tf.float32,
                               'total_distance' : tf.float32,
                               'num_samples' : tf.int32,
                               'scat_keep_prob' : tf.float32}
    members_dtypes = {}
    members_dtypes.update({'spec-' + k : paths_fields_dtypes[k] for k in paths_fields_dtypes})
    members_dtypes.update({'diff-' + k : paths_fields_dtypes[k] for k in paths_fields_dtypes})
    members_dtypes.update({'scat-' + k : paths_fields_dtypes[k] for k in paths_fields_dtypes})
    members_dtypes.update({'tmp-spec-' + k : tmp_paths_fields_dtypes[k] for k in tmp_paths_fields_dtypes})
    members_dtypes.update({'tmp-diff-' + k : tmp_paths_fields_dtypes[k] for k in tmp_paths_fields_dtypes})
    members_dtypes.update({'tmp-scat-' + k : tmp_paths_fields_dtypes[k] for k in tmp_paths_fields_dtypes})
    members_dtypes.update({'tmp-scat-' + k : tmp_paths_fields_dtypes[k] for k in tmp_paths_fields_dtypes})

    # Add the receiver position
    members_dtypes.update({'rx_pos' : tf.float32})

    # Add channel measurement
    members_dtypes.update({'h_meas' : tf.complex64})

    # Build dict of tensors
    # Deserializes the byte stream corresponding to each tensor
    features = {k : tf.io.FixedLenFeature([], tf.string, default_value = '') for k in members_dtypes}
    record = tf.io.parse_single_example(serialized_item, features)
    members_data = {k : tf.io.parse_tensor(record[k], out_type = members_dtypes[k]) for k in members_dtypes}

    # Builds the paths objects
    spec_data = {k[len('spec-'):] : members_data[k] for k in members_data if k.startswith('spec-')}
    diff_data = {k[len('diff-'):] : members_data[k] for k in members_data if k.startswith('diff-')}
    scat_data = {k[len('scat-'):] : members_data[k] for k in members_data if k.startswith('scat-')}
    tmp_spec_data = {k[len('tmp-spec-'):] : members_data[k] for k in members_data if k.startswith('tmp-spec-')}
    tmp_diff_data = {k[len('tmp-diff-'):] : members_data[k] for k in members_data if k.startswith('tmp-diff-')}
    tmp_scat_data = {k[len('tmp-scat-'):] : members_data[k] for k in members_data if k.startswith('tmp-scat-')}

    # Retrieve receiver position
    rx_pos = members_data['rx_pos']

    # Retrieve channel measurement
    h_meas = members_data['h_meas']

    # return spec_data

    return rx_pos, h_meas, spec_data, diff_data, scat_data, tmp_spec_data, tmp_diff_data, tmp_scat_data

def tensor_dicts_to_traced_paths(scene, tensor_dicts):
    """
    Creates Sionna `Paths` and `PathsTmpData` objects from dictionaries

    Input
    ------
    scene : Sionna.rt.Scene
        Scene

    tensor_dicts : `list` of `dict`
        List of dictionaries, as retrieved when iterating over a dataset
        of traced paths and using ``deserialize_paths_as_tensor_dicts()``
        to retrieve the data.

    Output
    -------
    spec_paths : Paths
        Specular paths

    diff_paths : Paths
        Diffracted paths

    scat_paths : Paths
        Scattered paths

    spec_paths_tmp : PathsTmpData
        Additional data required to compute the EM fields of the specular
        paths

    diff_paths_tmp : PathsTmpData
        Additional data required to compute the EM fields of the diffracted
        paths

    scat_paths_tmp : PathsTmpData
        Additional data required to compute the EM fields of the scattered
        paths
    """

    sources = tensor_dicts[0]['sources']
    targets = tensor_dicts[0]['targets']

    spec_paths = Paths(sources, targets, scene)
    spec_paths.from_dict(tensor_dicts[0])

    diff_paths = Paths(sources, targets, scene)
    diff_paths.from_dict(tensor_dicts[1])

    scat_paths = Paths(sources, targets, scene)
    scat_paths.from_dict(tensor_dicts[2])

    spec_tmp_paths = PathsTmpData(sources, targets, scene.dtype)
    spec_tmp_paths.from_dict(tensor_dicts[3])

    diff_tmp_paths = PathsTmpData(sources, targets, scene.dtype)
    diff_tmp_paths.from_dict(tensor_dicts[4])

    scat_tmp_paths = PathsTmpData(sources, targets, scene.dtype)
    scat_tmp_paths.from_dict(tensor_dicts[5])

    return (spec_paths, diff_paths, scat_paths, spec_tmp_paths, diff_tmp_paths, scat_tmp_paths)

def load_part_to_obj_dict(params_filename):
    with open(params_filename, "r") as openfile:
        part_to_obj_dict = json.load(openfile)
    return part_to_obj_dict

def load_obj_to_part_dict(params_filename):
    with open(params_filename, "r") as openfile:
        obj_to_part_dict = json.load(openfile)
    for object_name in obj_to_part_dict:
        obj_to_part_dict[object_name]["location"] = np.asarray(
            obj_to_part_dict[object_name]["location"], dtype=np.float64
        )  # shape is (3,)
        obj_to_part_dict[object_name]["dimension"] = np.asarray(
            obj_to_part_dict[object_name]["dimension"], dtype=np.float64
        )  # shape is (2, 3)
        obj_to_part_dict[object_name]["rotation"] = np.asarray(
            obj_to_part_dict[object_name]["rotation"], dtype=np.float64
        )  # shape is (3,)

    return obj_to_part_dict

def load_obj_part_dicts(scene_root_path):
    part_to_obj_path = os.path.join(scene_root_path, "part_to_obj.json")
    obj_to_part_path = os.path.join(scene_root_path, "obj_to_part.json")

    part_to_obj_dict = load_part_to_obj_dict(part_to_obj_path)
    obj_to_part_dict = load_obj_to_part_dict(obj_to_part_path)

    return part_to_obj_dict, obj_to_part_dict
