import tensorflow as tf
from keras.layers import Layer
from sionna import PI

from .utils import pos_enc, FcBlock


class NeuralInteraction(Layer):
    def __init__(self, num_layer, hidden_layer_dim, activation="exp"):
        super(NeuralInteraction, self).__init__()
        self.hidden_layer_dim = hidden_layer_dim
        self.activation = activation
        self.model = FcBlock(
            num_layer, hidden_layer_dim, 8, activation="relu", out_activation=None
        )

    def call(self, neural_mat_rep, geo_feature):
        geo_feature_enc = pos_enc(geo_feature, 4)

        model_input = tf.concat([neural_mat_rep, geo_feature, geo_feature_enc], axis=-1)
        model_output = self.model(model_input)

        mag, phase = tf.split(model_output, num_or_size_splits=2, axis=-1)
        if self.activation == "sigmoid":
            mag = tf.math.sigmoid(tf.clip_by_value(mag, -15.0, 15.0))

        else:
            mag = tf.math.exp(
                tf.clip_by_value(mag, tf.math.log(1e-7), tf.math.log(10.0))
            )
        phase = (tf.math.sigmoid(phase) - 0.5) * 2.0 * PI

        transfer_coeff = tf.complex(mag * tf.math.cos(phase), mag * tf.math.sin(phase))
        return transfer_coeff
