"""TensorFlow 1.x API for GraphSAGE under TensorFlow 2.x installs.

Import ``tf``, ``flags``, and ``FLAGS`` from this module instead of ``tensorflow``.
"""

from __future__ import division, print_function

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

flags = tf.flags
FLAGS = flags.FLAGS


def xavier_initializer():
    """Replacement for removed ``tf.contrib.layers.xavier_initializer``."""
    return tf.glorot_uniform_initializer()


def l2_regularizer(scale):
    """Replacement for ``tf.contrib.layers.l2_regularizer``."""
    if scale is None or scale == 0:
        return None

    def _regularizer(weights):
        return scale * tf.nn.l2_loss(weights)

    return _regularizer


def basic_lstm_cell(hidden_dim):
    """Replacement for ``tf.contrib.rnn.BasicLSTMCell``."""
    return tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
