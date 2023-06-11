import tensorflow as tf


class DiceLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true_c = 1.0-tf.cast(y_true, tf.float32)
        y_pred_c = 1.0-tf.cast(y_pred, tf.float32)
        numerator = 2 * tf.reduce_sum(y_true_c * y_pred_c)
        denominator = tf.reduce_sum(y_true_c + y_pred_c)

        return 1 - numerator / denominator
