import tensorflow as tf


class AlexNetHelper:
    @staticmethod
    def instintiate_weights(key, shape):
        return tf.get_variable(key,
                               shape=shape,
                               initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                          mode='FAN_IN',
                                                                                          uniform=False,
                                                                                          seed=None,
                                                                                          dtype=tf.float32))

    @staticmethod
    def instintiate_bias(key, shape):
        return tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=shape))
