import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.input_spec import InputSpec

##################################################################################
##################################################################################
##################################################################################
##################################################################################

class MSMM_Layer(layers.Layer):

    def __init__(self, nfeat=1):
        super(MSMM_Layer, self).__init__()
        self.nfeat = nfeat
        alpha_init = tf.random_normal_initializer()
        beta_init = tf.random_normal_initializer()
        self.alpha = tf.Variable(
            initial_value=alpha_init(
                shape=(4 * self.nfeat,),
                dtype='float32',
            ),
            trainable=True,
        )
        self.beta = tf.Variable(
            initial_value=beta_init(
                shape=(4 * self.nfeat,),
                dtype='float32',
            ),
            trainable=True,
        )

    def call(self, inputs):
        res = []
        for i in range(self.nfeat):
            # (weighted) Mean
            res.append(
                self.alpha[4*i+0] * tf.reduce_sum(
                     inputs[:, (i+1)::(self.nfeat + 1)]
                    * inputs[:, ::(self.nfeat + 1)],
                    axis=-1,
                )
                / tf.reduce_sum(inputs[:, ::(self.nfeat + 1)], axis=-1)
                + self.beta[4*i+0]
            )
            # Sum
            res.append(
                self.alpha[4*i+1] * tf.reduce_sum(
                     inputs[:, (i+1)::(self.nfeat + 1)]
                    * inputs[:, ::(self.nfeat + 1)],
                    axis=-1,
                )
                + self.beta[4*i+1]
            )
            # Min
            res.append(
                self.alpha[4*i+2] * tf.reduce_min(
                     inputs[:, (i+1)::(self.nfeat + 1)]
                    * inputs[:, ::(self.nfeat + 1)],
                    axis=-1,
                )
                + self.beta[4*i+2]
            )
            # Max
            res.append(
                self.alpha[4*i+3] * tf.reduce_max(
                     inputs[:, (i+1)::(self.nfeat + 1)]
                    * inputs[:, ::(self.nfeat + 1)],
                    axis=-1,
                )
                + self.beta[4*i+3]
            )
        return tf.stack(res, axis=1)

##################################################################################
##################################################################################
##################################################################################
##################################################################################

class Maxi_Layer(layers.Layer):

    def __init__(self, nfeat=1, ndense=[], nMSMM=1):
        super(Maxi_Layer, self).__init__()
        self.nfeat = nfeat
        self.nMSMM = nMSMM
        self.denses = []
        for n in ndense:
            self.denses.append(Dense(n, activation='relu'))
        alpha_init = tf.random_normal_initializer()
        beta_init = tf.random_normal_initializer()
        self.alpha = tf.Variable(
            initial_value=alpha_init(
                shape=(4 * self.nfeat * self.nMSMM,),
                dtype='float32',
            ),
            trainable=True,
        )
        self.beta = tf.Variable(
            initial_value=beta_init(
                shape=(4 * self.nfeat * self.nMSMM,),
                dtype='float32',
            ),
            trainable=True,
        )

    def call(self, inputs):
        res = []
        for i in range(self.nfeat):
            # (weighted) Mean
            res.append(
                self.alpha[4*i+0] * tf.reduce_sum(
                     inputs[:, (i+1)::(self.nfeat + 1)]
                    * inputs[:, ::(self.nfeat + 1)],
                    axis=-1,
                )
                / tf.reduce_sum(inputs[:, ::(self.nfeat + 1)], axis=-1)
                + self.beta[4*i+0]
            )
            # Sum
            res.append(
                self.alpha[4*i+1] * tf.reduce_sum(
                     inputs[:, (i+1)::(self.nfeat + 1)]
                    * inputs[:, ::(self.nfeat + 1)],
                    axis=-1,
                )
                + self.beta[4*i+1]
            )
            # Min
            res.append(
                self.alpha[4*i+2] * tf.reduce_min(
                     inputs[:, (i+1)::(self.nfeat + 1)]
                    * inputs[:, ::(self.nfeat + 1)],
                    axis=-1,
                )
                + self.beta[4*i+2]
            )
            # Max
            res.append(
                self.alpha[4*i+3] * tf.reduce_max(
                     inputs[:, (i+1)::(self.nfeat + 1)]
                    * inputs[:, ::(self.nfeat + 1)],
                    axis=-1,
                )
                + self.beta[4*i+3]
            )
        return tf.stack(res, axis=1)


class ShuffleRepeatVector(Layer):
    """Repeats the input n times.
    Example:
    ```python
    model = Sequential()
    model.add(Dense(32, input_dim=32))
    # now: model.output_shape == (None, 32)
    # note: `None` is the batch dimension
    model.add(RepeatVector(3))
    # now: model.output_shape == (None, 3, 32)
    ```
    Arguments:
        n: Integer, repetition factor.
    Input shape:
        2D tensor of shape `(num_samples, features)`.
    Output shape:
        3D tensor of shape `(num_samples, n, features)`.
    """

    def __init__(self, n, **kwargs):
        super(ShuffleRepeatVector, self).__init__(**kwargs)
        self.n = n
        self.input_spec = InputSpec(ndim=2)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return tensor_shape.TensorShape([input_shape[0], self.n, input_shape[1]])

    def call(self, inputs):
        return K.repeat(inputs, self.n)

    def get_config(self):
        config = {'n': self.n}
        base_config = super(ShuffleRepeatVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))