import numpy as np
from scipy.spatial.transform import Rotation as R

import tensorflow as tf
from keras.layers import Layer

from .utils import pos_enc
from .neural_object import NeuralObject
from .neural_interaction import NeuralInteraction


class NeuralScene(Layer):
    def __init__(
        self,
        scene,
        part_to_obj_dict,
        obj_to_part_dict,
        pos_encoding_size=10,
        learn_scattering=True,
        verbose=False,
    ):
        super(NeuralScene, self).__init__()

        self.num_objects = len(obj_to_part_dict)
        self.num_parts = len(scene.objects)

        self.obj_to_part_dict = obj_to_part_dict
        self.part_to_obj_dict = part_to_obj_dict

        self.frequency = scene.frequency
        self.pos_encoding_size = pos_encoding_size
        self.learn_scattering = learn_scattering
        self.verbose = verbose

        self.get_part_to_obj_info(scene)
        self.get_obj_trans_info()

        self.mat_rep_dim = 48

        self.neural_objects = []
        self.neural_refl = NeuralInteraction(
            num_layer=3, hidden_layer_dim=128, activation="sigmoid"
        )
        self.neural_diffr = NeuralInteraction(num_layer=5, hidden_layer_dim=128)
        self.neural_scat = NeuralInteraction(num_layer=3, hidden_layer_dim=128)
        self.neural_interaction = [
            self.neural_refl,
            self.neural_diffr,
            self.neural_scat,
        ]

    def get_part_to_obj_info(self, scene):
        # self.num_parts+1 because the last part or obj is a placeholder
        # for invalid objects whose idx is -1
        part_to_obj = np.zeros((self.num_parts + 1,), dtype=np.int64)

        for part_name in scene.objects.keys():
            part_idx = scene.objects[part_name].object_id
            object_idx = self.part_to_obj_dict[part_name]["object_idx"]
            part_to_obj[part_idx] = object_idx
        # object_idx 0~self.num_objects-1 is for valid objects
        # object_idx=self.num_objects is the placeholder for the invalid object
        part_to_obj[self.num_parts] = self.num_objects
        self.part_to_obj = tf.constant(part_to_obj, dtype=tf.int64)

    def get_obj_trans_info(self):
        object_dim_all = []
        object_rot_mtx_all = []
        object_location_all = []
        object_rot_mtx_all = []
        for object_name in self.obj_to_part_dict.keys():
            obj = self.obj_to_part_dict[object_name]
            object_loc = obj["location"]
            object_rot = obj["rotation"]

            # object dimensions
            object_dim = np.asarray(obj["dimension"])
            object_dim = np.reshape(object_dim, (2, 3))
            object_dim_all.append(
                object_dim
            )  # this will be used for normalizing the interaction position

            # object location
            object_location = np.asarray(object_loc)
            object_location = np.reshape(object_location, (3,))
            object_location_all.append(object_location)

            # object rotation matrix
            rot_mtx = R.from_euler(
                "xyz", [object_rot[0], object_rot[1], object_rot[2]], degrees=False
            ).as_matrix()

            # we need the inverse rotation to transform the interaction position back to the object's local coordinate
            rot_mtx = np.linalg.inv(rot_mtx)
            object_rot_mtx_all.append(rot_mtx)

        # placeholder for the last invalid object
        object_dim_all.append(np.zeros_like(object_dim_all[-1]))
        object_location_all.append(np.zeros_like(object_location_all[-1]))
        object_rot_mtx_all.append(np.zeros_like(object_rot_mtx_all[-1]))
        # stack
        object_dim_all = np.stack(object_dim_all, 0)
        object_location_all = np.stack(object_location_all, 0)
        object_rot_mtx_all = np.stack(object_rot_mtx_all, 0)

        object_scale_all = object_dim_all[:, 1, :] - object_dim_all[:, 0, :]
        object_scale_all = np.amax(np.squeeze(object_scale_all), -1)
        # placeholder for the last invalid object to have scale=1
        object_scale_all[..., -1] = 1

        self.object_dim_all = tf.constant(object_dim_all, dtype=tf.float32)
        self.object_scale_all = tf.constant(object_scale_all, dtype=tf.float32)
        self.object_location_all = tf.constant(object_location_all, dtype=tf.float32)
        self.object_rot_mtx_all = tf.constant(object_rot_mtx_all, dtype=tf.float32)

    def to_local_direction(self, obj_idx, sionna_obj_idx, x):
        data_shape = sionna_obj_idx.shape.as_list()
        rot = tf.gather(self.object_rot_mtx_all, obj_idx, axis=0)
        rot = tf.reshape(rot, data_shape + [1, 3, 3])

        x_u = rot @ tf.expand_dims(x, -1)
        x_u = tf.squeeze(x_u, axis=-1)

        return x_u

    def to_unit_bbox(self, obj_idx, sionna_obj_idx, x):
        data_shape = sionna_obj_idx.shape.as_list()

        dim = tf.gather(self.object_dim_all, obj_idx)
        dim = tf.reshape(dim, data_shape + [2, 3])

        scale = tf.gather(self.object_scale_all, obj_idx)
        scale = tf.reshape(scale, data_shape + [1])
        loc = tf.gather(self.object_location_all, obj_idx, axis=0)
        loc = tf.reshape(loc, data_shape + [3])
        rot = tf.gather(self.object_rot_mtx_all, obj_idx, axis=0)
        rot = tf.reshape(rot, data_shape + [3, 3])

        x_u = x - loc
        x_u = rot @ tf.expand_dims(x_u, -1)
        x_u = tf.squeeze(x_u, axis=-1)
        # object is centered in blender such that min_z=0.
        # here we shift the object to the true center, that is min_z = -max_z
        x_u = x_u - ((dim[..., 1, :] + dim[..., 0, :]) / 2)

        x_u = x_u / scale * 2.0  # x_u is normalized to (-1, 1)

        return x_u

    def build(self, input_shape):
        for _ in range(self.num_objects + 1):
            # the last model is a placeholder for invalid objects whose idx is -1
            # Build the neural network
            self.neural_objects.append(NeuralObject(3, 64, 16, 2, 64, self.mat_rep_dim))

    def call(self, sionna_obj_idx, pos, geo_feat, interaction_type):
        sionna_obj_idx_flatten = tf.reshape(sionna_obj_idx, (-1,))
        sionna_obj_idx_flatten = tf.where(
            sionna_obj_idx_flatten != -1,
            sionna_obj_idx_flatten,
            tf.zeros_like(sionna_obj_idx_flatten) + self.num_parts,
        )

        obj_idx = tf.gather(self.part_to_obj, sionna_obj_idx_flatten)

        # Fit to unit cube
        pos = self.to_unit_bbox(obj_idx, sionna_obj_idx, pos)

        # Encode position
        # [..., 3*2*pos_encoding_size]
        enc_pos = pos_enc(pos, self.pos_encoding_size)
        data_shape = enc_pos.shape[:-1]

        enc_pos = tf.reshape(enc_pos, [-1, enc_pos.shape[-1]])
        geo_feat = tf.reshape(geo_feat, [-1, geo_feat.shape[-1]])

        enc_pos_all_obj = tf.TensorArray(
            tf.float32,
            size=self.num_objects + 1,
            infer_shape=False,
            element_shape=tf.TensorShape([None, enc_pos.shape[-1]]),
        )
        geo_feat_all_obj = tf.TensorArray(
            tf.float32,
            size=self.num_objects + 1,
            infer_shape=False,
            element_shape=tf.TensorShape([None, geo_feat.shape[-1]]),
        )
        data_all_idx = tf.TensorArray(
            tf.int64,
            size=self.num_objects + 1,
            infer_shape=False,
            element_shape=tf.TensorShape(
                [
                    None,
                ]
            ),
        )

        for i in range(self.num_objects + 1):
            data_idx = tf.reshape(tf.where(obj_idx == i), (-1,))
            data_all_idx = data_all_idx.write(i, data_idx)

            enc_pos_one_obj = tf.gather(enc_pos, data_idx)
            enc_pos_all_obj = enc_pos_all_obj.write(i, enc_pos_one_obj)

            geo_feat_one_obj = tf.gather(geo_feat, data_idx)
            geo_feat_all_obj = geo_feat_all_obj.write(i, geo_feat_one_obj)

        data_all_idx = data_all_idx.concat()
        data_all_idx = tf.expand_dims(data_all_idx, -1)

        transfer_coeff = tf.TensorArray(
            tf.complex64,
            size=self.num_objects + 1,
            infer_shape=False,
            element_shape=tf.TensorShape([None, 4]),
        )
        neural_mat_rep_all = []
        for i in range(self.num_objects + 1):
            pos_input = tf.identity(enc_pos_all_obj.read(i))
            geo_input = tf.identity(geo_feat_all_obj.read(i))

            neural_mat_rep = self.neural_objects[i](pos_input)
            if tf.size(neural_mat_rep) == 0:
                neural_mat_rep = neural_mat_rep * 0.0

            neural_mat_rep_all.append(neural_mat_rep)

            transfer_coeff_ = self.neural_interaction[interaction_type](
                neural_mat_rep, geo_input
            )
            transfer_coeff = transfer_coeff.write(i, transfer_coeff_)
        transfer_coeff = transfer_coeff.concat()
        transfer_coeff = tf.scatter_nd(
            data_all_idx,
            transfer_coeff,
            shape=tf.shape(transfer_coeff, out_type=tf.int64),
        )
        transfer_coeff = tf.reshape(
            transfer_coeff, data_shape.as_list() + [transfer_coeff.shape[-1]]
        )
        transfer_mtx = tf.split(transfer_coeff, num_or_size_splits=2, axis=-1)
        transfer_mtx = tf.stack(transfer_mtx, -1)

        if self.verbose:
            neural_mat_rep_all = tf.concat(neural_mat_rep_all, axis=0)
            return transfer_mtx, neural_mat_rep_all

        return transfer_mtx
