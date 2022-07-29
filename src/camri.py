import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
import numpy as np


class CAMRI(Layer):
    def __init__(self, important_class, m, s, n_classes, regularizer=None, **kwargs):
        super(CAMRI, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.regularizer = regularizers.get(regularizer)

        # define class-sensitive additive angular margin (CAMRI)
        margin = np.zeros((1, n_classes))
        margin[0, important_class] = m
        self.m = tf.cast(margin, 'float32')

    def build(self, input_shape):
        super(CAMRI, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        x = tf.nn.l2_normalize(x, axis=1)  # normalize feature
        W = tf.nn.l2_normalize(self.W, axis=0)  # normalize weights

        logits = x @ W  # calc cos
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))  # calc theta
        target_logits = tf.cos(theta + self.m)  # add CAMRI

        logits = logits * (1 - y) + target_logits * y
        logits *= self.s  # scaling

        return logits

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)
