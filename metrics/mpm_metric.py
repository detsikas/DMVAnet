import tensorflow as tf
import numpy as np
import cv2


class MPMMetric(tf.keras.metrics.Metric):

    def __init__(self, name='mpm', **kwargs):
        super(MPMMetric, self).__init__(name=name, **kwargs)
        self.mpm = self.add_weight(name='mpm', initializer='zeros')
        self.count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        fp = np.zeros(y_true.shape)
        fp[(y_pred == 0) & (y_true == 1)] = 1
        fn = np.zeros(y_true.shape)
        fn[(y_pred == 1) & (y_true == 0)] = 1

        kernel = np.ones((3, 3), dtype=np.uint8)
        im_dil = cv2.erode(y_true, kernel)
        im_gtb = y_true - im_dil
        im_gtbd = cv2.distanceTransform(1 - im_gtb, cv2.DIST_L2, 3)
        nd = im_gtbd.sum()
        im_dn = im_gtbd.copy()
        im_dn[fn == 0] = 0
        dn = np.sum(im_dn)
        mpfn = dn / nd

        im_dp = im_gtbd.copy()
        im_dp[fp == 0] = 0
        dp = np.sum(im_dp)
        mpfp = dp / nd
        values = (mpfp + mpfn) / 2

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        self.mpm.assign_add(values)
        self.count += 1

    def result(self):
        return self.mpm/self.count
