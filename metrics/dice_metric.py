import tensorflow as tf


class DiceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='binary_dice', **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.dice = self.add_weight(name='dice', initializer='zeros')
        self.count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = 1.0-tf.cast(y_true, tf.float32)
        y_pred = 1.0-tf.cast(y_pred, tf.float32)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        values = 1 - numerator / denominator
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        self.dice.assign(values)
        self.count += 1

    def result(self):
        return self.dice/self.count
