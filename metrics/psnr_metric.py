import tensorflow as tf


class PSNRMetric(tf.keras.metrics.Metric):
    def __init__(self, name='psnr', **kwargs):
        super(PSNRMetric, self).__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name='psnr', initializer='zeros')
        self.count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.image.psnr(y_true, y_pred, max_value=1)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        self.psnr.assign_add(values)
        self.count += 1

    def result(self):
        return self.psnr/self.count
