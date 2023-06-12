import tensorflow as tf


class F1Metric(tf.keras.metrics.Metric):
    def __init__(self, name='binary_fm', **kwargs):
        super(F1Metric, self).__init__(name=name, **kwargs)
        self.fm = self.add_weight(name='fm', initializer='zeros')
        self.count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.bool), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.bool), [-1])

        cfm = tf.cast(tf.math.confusion_matrix(y_true, y_pred), self.dtype)
        # tn = cfm[0, 0]
        tp = cfm[0, 0]
        fp = cfm[1, 0]
        fn = cfm[0, 1]

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, tp.shape)
            tp = tf.multiply(tp, sample_weight)
            fp = tf.multiply(fp, sample_weight)
            fn = tf.multiply(fn, sample_weight)

        rec_values = tp / (tp + fn)
        prec_values = tp / (tp + fp)

        fm_values = (2 * rec_values * prec_values) / (rec_values + prec_values)

        self.fm.assign_add(fm_values)
        self.count += 1

    def result(self):
        return self.fm/self.count
