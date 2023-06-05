import tensorflow as tf
import numpy as np
from scipy import ndimage as ndi


class PseudoF1Metric(tf.keras.metrics.Metric):

    G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
                         0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
                         1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                         0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
                         0, 0, 0], dtype=np.bool)

    G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                          1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                          0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                          1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                          0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0], dtype=np.bool)

    def __init__(self, name='pseudo f1', **kwargs):
        super(PseudoF1Metric, self).__init__(name=name, **kwargs)
        self.pf1 = self.add_weight(name='pseudo_f1', initializer='zeros')
        self.p_precison = self.add_weight(
            name='pseudo_precision', initializer='zeros')
        self.p_recall = self.add_weight(
            name='pseudo_recallZ', initializer='zeros')

        self.count = 0

    def bwmorph_thin(self, image, n_iter=None):
        # check parameters
        if n_iter is None:
            n = -1
        elif n_iter <= 0:
            raise ValueError('n_iter must be > 0')
        else:
            n = n_iter

        # check that we have a 2d binary image, and convert it
        # to uint8
        skel = np.array(image).astype(np.uint8)

        if skel.ndim != 2:
            raise ValueError('2D array required')
        if not np.all(np.in1d(image.flat, (0, 1))):
            raise ValueError('Image contains values other than 0 and 1')

        # neighborhood mask
        mask = np.array([[8, 4, 2],
                        [16, 0, 1],
                        [32, 64, 128]], dtype=np.uint8)

        # iterate either 1) indefinitely or 2) up to iteration limit
        while n != 0:
            before = np.sum(skel)  # count points before thinning

            # for each subiteration
            for lut in [PseudoF1Metric.G123_LUT, PseudoF1Metric.G123P_LUT]:
                # correlate image with neighborhood mask
                N = ndi.correlate(skel, mask, mode='constant', cval=0)
                # take deletion decision from this subiteration's LUT
                D = np.take(lut, N)
                # perform deletion
                skel[D] = 0

            after = np.sum(skel)  # coint points after thinning

            if before == after:
                # iteration had no effect: finish
                break

            # count down to iteration limit (or endlessly negative)
            n -= 1

        return skel.astype(np.bool)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.cast(y_true, tf.bool)
        # y_pred = tf.cast(y_pred, tf.bool)

        inverse_gt_image = 1 - y_true
        skeleton_gt_image = self.bwmorph_thin(inverse_gt_image)
        skeleton_gt_image = skeleton_gt_image.astype('uint8')
        skeleton_gt_image = 1 - skeleton_gt_image
        temp_tp = np.multiply((1 - y_pred), (1 - y_true))
        temp_fp = (1 - y_pred) * y_true
        temp_fn = y_pred * (1 - y_true)
        temp_tn = y_pred * y_true
        count_tp = np.sum(temp_tp)
        count_fp = np.sum(temp_fp)
        count_fn = np.sum(temp_fn)
        count_tn = np.sum(temp_tn)
        temp_p = count_tp / (count_fp + count_tp)
        temp_skl_tp = (1 - y_pred) * (1 - skeleton_gt_image)
        temp_skl_fp = (1 - y_pred) * skeleton_gt_image
        temp_skl_fn = y_pred * (1 - skeleton_gt_image)
        temp_skl_tn = y_pred * skeleton_gt_image
        count_skl_tp = np.sum(temp_skl_tp)
        count_skl_fp = np.sum(temp_skl_fp)
        count_skl_fn = np.sum(temp_skl_fn)
        count_skl_tn = np.sum(temp_skl_tn)
        temp_pseudo_p = count_skl_tp / (count_skl_fp + count_skl_tp)
        temp_pseudo_r = count_skl_tp / (count_skl_fn + count_skl_tp)
        temp_pseudo_f = 2.0 * temp_p * temp_pseudo_r / (temp_p + temp_pseudo_r)
        self.pf1.assign_add(temp_pseudo_f)
        self.p_precison.assign_add(temp_pseudo_p)
        self.p_recall.assign_add(temp_pseudo_r)
        self.count += 1

    def result(self):
        return {'p_f1': self.pf1/self.count, 'p_precision': self.p_precison/self.count, 'p_recall': self.p_recall/self.count}
