import tensorflow as tf
import common.metrics as metrics


# Inputs are [0.0,1.0]
def mse(gt_image, predicted_image):
    m = tf.keras.metrics.MeanSquaredError()
    m.update_state(gt_image, predicted_image)
    return m.result().numpy()


def bce(gt_image, predicted_image):
    b = tf.keras.metrics.BinaryCrossentropy()
    b.update_state(gt_image, predicted_image)
    return b.result().numpy()


def bac(gt_image, predicted_image):
    bac_func = tf.keras.metrics.BinaryAccuracy()
    bac_func.update_state(gt_image, predicted_image)
    return bac_func.result().numpy()


def psnr(gt_image, predicted_image):
    exp_gt_image = tf.expand_dims(gt_image, axis=0)
    exp_predicted_image = tf.expand_dims(predicted_image, axis=0)
    exp_predicted_image = tf.expand_dims(exp_predicted_image, axis=3)
    psnr_tf = metrics.PSNRMetric()
    psnr_tf.update_state(tf.cast(exp_gt_image, tf.float32),
                         tf.cast(exp_predicted_image, tf.float32))
    return psnr_tf.result().numpy()


def dice(gt_image, predicted_image):
    exp_gt_image = tf.expand_dims(gt_image, axis=0)
    exp_predicted_image = tf.expand_dims(predicted_image, axis=0)
    exp_predicted_image = tf.expand_dims(exp_predicted_image, axis=3)
    dice_tf = metrics.DiceMetric()
    dice_tf.update_state(tf.cast(exp_gt_image, tf.float32), tf.cast(exp_predicted_image, tf.float32))
    return dice_tf.result().numpy()

