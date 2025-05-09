import tensorflow as tf
from keras.layers import Layer, Dense
from sionna import PI
from sionna.utils import expand_to_rank, flatten_last_dims


def pos_enc(x, encoding_size):
    x = tf.expand_dims(x, axis=-1)
    indices = tf.range(encoding_size, dtype=tf.float32)
    indices = expand_to_rank(indices, tf.rank(x), 0)
    enc = tf.pow(2.0, indices) * PI * x
    enc_cos = tf.math.cos(enc)
    enc_sin = tf.math.sin(enc)
    enc = tf.concat([enc_cos, enc_sin], axis=-1)
    enc = flatten_last_dims(enc, 2)

    return enc


class FcBlock(Layer):
    def __init__(
        self,
        num_layer,
        hidden_layer_dim,
        out_dim,
        activation="relu",
        out_activation=None,
    ):
        super(FcBlock, self).__init__()
        self.num_layer = num_layer
        self.hidden_layer_dim = hidden_layer_dim
        self.out_dim = out_dim
        self.activation = activation
        self.out_activation = out_activation
        self.layers = []

    def build(self, input_shape):
        for _ in range(self.num_layer - 1):
            self.layers.append(Dense(self.hidden_layer_dim, self.activation))
        self.layers.append(Dense(self.out_dim, self.out_activation))

    def call(self, model_input):
        for layer in self.layers:
            model_input = layer(model_input)
        return model_input


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate, end_iter, warm_up_end=2000, alpha=0.05):
        super().__init__()
        self.learning_rate = learning_rate
        self.end_iter = end_iter
        self.warm_up_end = warm_up_end
        self.alpha = alpha

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm_up_end = tf.cast(self.warm_up_end, tf.float32)
        end_iter = tf.cast(self.end_iter, tf.float32)
        alpha = tf.cast(self.alpha, tf.float32)

        def warmup():
            return step / warm_up_end

        def cosine_decay():
            progress = tf.clip_by_value(
                (step - warm_up_end) / (end_iter - warm_up_end), 0.0, 1.0
            )
            return (tf.math.cos(PI * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        factor = tf.cond(step < warm_up_end, warmup, cosine_decay)
        return self.learning_rate * factor
