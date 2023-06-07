import tensorflow as tf


class NRMMetric(tf.keras.metrics.Metric):
    def __init__(self, name='nrm', **kwargs):
        super(NRMMetric, self).__init__(name=name, **kwargs)
        self.nrm = self.add_weight(name='nrm', initializer='zeros')
        self.count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        cfm = tf.math.confusion_matrix(y_true, y_pred)
        tn = cfm[0, 0]
        tp = cfm[1, 1]
        fp = cfm[0, 1]
        fn = cfm[1, 0]

        nrfn = fn / (fn + tp)
        nrfp = fp / (fp + tn)
        values = (nrfn + nrfp) / 2.0

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        self.nrm.assign_add(values)
        self.count += 1

    def result(self):
        return self.nrm/self.count