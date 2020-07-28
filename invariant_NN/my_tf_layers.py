import tensorflow as tf
from tensorflow.keras import layers

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
