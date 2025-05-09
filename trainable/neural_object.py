import tensorflow as tf
from keras.layers import Layer
from .utils import FcBlock


class NeuralObject(Layer):
    def __init__(
        self,
        num_layer_enc,
        hidden_layer_dim_enc,
        enc_dim,
        num_layer_dec,
        hidden_layer_dim_dec,
        nerual_mat_dim,
    ):
        super(NeuralObject, self).__init__()
        self.num_layer_enc = num_layer_enc
        self.hidden_layer_dim_enc = hidden_layer_dim_enc
        self.enc_dim = enc_dim
        self.num_layer_dec = num_layer_dec
        self.hidden_layer_dim_dec = hidden_layer_dim_dec
        self.nerual_mat_dim = nerual_mat_dim

        self.encoder = FcBlock(
            num_layer_enc,
            hidden_layer_dim_enc,
            enc_dim,
            activation="relu",
            out_activation="relu",
        )
        self.decoder = FcBlock(
            num_layer_dec,
            hidden_layer_dim_dec,
            nerual_mat_dim,
            activation="relu",
            out_activation="sigmoid",
        )

    def call(self, ip_pos_local):
        encoding = self.encoder(ip_pos_local)
        encoding = tf.concat([encoding, ip_pos_local], axis=-1)
        neural_mat_rep = self.decoder(encoding)
        return neural_mat_rep
