import tensorflow as tf

'''
J. Pastor-Pellicer, F. Zamora-Mart ́ınez, S. Espa ̃na-Boquera, M. J. Castro-
Bleda, F-measure as the error function to train neural networks, in: Inter-
national Work-Conference on Artificial Neural Networks, Springer, 2013,
pp. 376–384
'''
class MicroFMLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        beta = 1.0
        beta2 = beta ** 2.0
        y_pred = tf.squeeze(y_pred, axis=3)
        top = tf.math.reduce_sum(y_true * y_pred)
        bot = beta2 * tf.math.reduce_sum(y_true) + tf.math.reduce_sum(y_pred)
        return -(1.0 + beta2) * top / bot


class ComboLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        dice_loss = DiceLoss()
        micro_fm_loss = MicroFMLoss()
        dice_loss_result = dice_loss(y_true, y_pred)
        micro_fm_loss_result = micro_fm_loss(y_true, y_pred)
        return dice_loss_result+micro_fm_loss_result


class ComboLossWeighted(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        dice_loss = DiceLoss()
        micro_fm_loss = MicroFMLoss()
        dice_loss_result = dice_loss(y_true, y_pred)
        micro_fm_loss_result = micro_fm_loss(y_true, y_pred)
        return dice_loss_result+10.0*(1+micro_fm_loss_result)


class DiceLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred, axis=3)
        y_true_c = 1.0-tf.cast(y_true, tf.float32)
        y_pred_c = 1.0-tf.cast(y_pred, tf.float32)
        numerator = 2 * tf.reduce_sum(y_true_c * y_pred_c)
        denominator = tf.reduce_sum(y_true_c + y_pred_c)

        return 1 - numerator / denominator


class DiceLoss3(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true_c = tf.cast(y_true, tf.float32)
        y_pred_c = tf.cast(y_pred, tf.float32)
        numerator = 2 * tf.reduce_sum(y_true_c * y_pred_c)
        denominator = tf.reduce_sum(y_true_c + y_pred_c)

        return 1 - numerator / denominator


class DiceMSELoss(tf.keras.losses.Loss):
    def __init__(self, alpha):
        super(DiceMSELoss, self).__init__()
        self.__alpha = alpha

    @ tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        dice_loss = DiceLoss()
        dice_loss_result = dice_loss(y_true, y_pred)
        return dice_loss_result + self.__alpha*tf.keras.losses.MeanSquaredError()(y_true,y_pred)


class DicePSNRLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        dice_loss = DiceLoss()
        dice_loss_result = dice_loss(y_true, y_pred)
        psnr_loss = InvPSNRLoss()

        return dice_loss_result + 10.0*psnr_loss(y_true, y_pred)


class InvPSNRLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred, axis=3)
        return 1.0/tf.image.psnr(y_true, y_pred, max_val=1.0)


class NRMLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        _cfm = tf.math.confusion_matrix(y_true, y_pred)
        _tn = _cfm[0, 0]
        _tp = _cfm[1, 1]
        _fp = _cfm[0, 1]
        _fn = _cfm[1, 0]

        _nrfn = _fn / (_fn + _tp)
        _nrfp = _fp / (_fp + _tn)
        _nrm = (_nrfn + _nrfp) / 2.0
        return _nrm

'''
https://arxiv.org/pdf/2006.14822.pdf
Source https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
'''
class FocalLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred, axis=3)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
                                      alpha=0.25, gamma=2.0, y_pred=y_pred)

        return tf.reduce_mean(loss)

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b


class TverskyLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return 1 - self.tversky_index(y_true, y_pred)

    def tversky_index(self, y_true, y_pred):
        y_true_pos = tf.reshape(y_true, [-1])
        y_pred_pos = tf.reshape(y_pred, [-1])
        true_pos = tf.math.reduce_sum(y_true_pos * y_pred_pos)
        false_neg = tf.math.reduce_sum(y_true_pos * (1 - y_pred_pos))
        false_pos = tf.math.reduce_sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + 1.0) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + 1.0)


class FocalTverskyLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        pt_1 = self.tversky_index(y_true, y_pred)
        gamma = 0.75
        return tf.math.pow((1 - pt_1), gamma)

    def tversky_index(self, y_true, y_pred):
        y_true_pos = tf.reshape(y_true, [-1])
        y_pred_pos = tf.reshape(y_pred, [-1])
        true_pos = tf.math.reduce_sum(y_true_pos * y_pred_pos)
        false_neg = tf.math.reduce_sum(y_true_pos * (1 - y_pred_pos))
        false_pos = tf.math.reduce_sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + 1.0) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + 1.0)


class LogCoshDiceLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        dice_loss = DiceLoss()
        x = dice_loss(y_true, y_pred)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

