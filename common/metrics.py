import numpy as np
import tensorflow as tf
import common.losses as losses
from skimage.util.shape import view_as_windows
from scipy import ndimage as ndi

try:
    import cv2
except ModuleNotFoundError:
    pass
import numpy.polynomial.polynomial as poly

# Reference https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05


def threshold_image(img):
    local_image = np.copy(img)
    local_image[local_image < 0.5] = 0
    local_image[local_image >= 0.5] = 1
    return local_image.astype('uint8')


def threshold_image_otsu(img):
    local_image = np.copy(img)
    ret2,th2 = cv2.threshold((local_image*255).astype('uint8'),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print('Threshold: {}'.format(ret2))
    return th2/255


'''
from sklearn.mixture import GaussianMixture
global_th = None
def threshold_image_gmm(img):
    global global_th
    local_image = np.copy(img)
    if global_th is None:
        flat_image = local_image.flatten()
        gmm = GaussianMixture(n_components=2, tol=0.000001)
        gmm.fit(np.expand_dims(flat_image, 1))
        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()
        a = 1.0/variances[0]-1.0/variances[1]
        b = 2.0*(means[1]/variances[1]-means[0]/variances[0])
        d = 2.0*np.log(np.sqrt(variances[1]/variances[0])*weights[0]/weights[1])
        c = means[0]*means[0]/variances[0]-means[1]*means[1]/variances[1]-d
        roots = poly.polyroots((c, b, a))
        print(roots)
        global_th = roots[np.nonzero(roots<1)]
    print(global_th)
    local_image[local_image < global_th] = 0
    local_image[local_image >= global_th] = 1
    return local_image.astype('uint8')
'''

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


def bwmorph_thin(image, n_iter=None):
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
        for lut in [G123_LUT, G123P_LUT]:
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


def pfmeasure(gt_image, predicted_image):
    inverse_gt_image = 1 - gt_image
    skeleton_gt_image = bwmorph_thin(inverse_gt_image)
    skeleton_gt_image = skeleton_gt_image.astype('uint8')
    skeleton_gt_image = 1 - skeleton_gt_image
    temp_tp = np.multiply((1 - predicted_image),(1 - gt_image))
    temp_fp = (1 - predicted_image) * gt_image
    temp_fn = predicted_image * (1 - gt_image)
    temp_tn = predicted_image * gt_image
    count_tp = np.sum(temp_tp)
    count_fp = np.sum(temp_fp)
    count_fn = np.sum(temp_fn)
    count_tn = np.sum(temp_tn)
    temp_p = count_tp / (count_fp + count_tp)
    temp_skl_tp = (1 - predicted_image) * (1 - skeleton_gt_image)
    temp_skl_fp = (1 - predicted_image) * skeleton_gt_image
    temp_skl_fn = predicted_image * (1 - skeleton_gt_image)
    temp_skl_tn = predicted_image * skeleton_gt_image
    count_skl_tp = np.sum(temp_skl_tp)
    count_skl_fp = np.sum(temp_skl_fp)
    count_skl_fn = np.sum(temp_skl_fn)
    count_skl_tn = np.sum(temp_skl_tn)
    temp_pseudo_p = count_skl_tp / (count_skl_fp + count_skl_tp)
    temp_pseudo_r = count_skl_tp / (count_skl_fn + count_skl_tp)
    temp_pseudo_f = 2.0 * temp_p * temp_pseudo_r / (temp_p + temp_pseudo_r)
    return temp_pseudo_f, temp_pseudo_p, temp_pseudo_r


def confusion_matrix(gt_image, predicted_image):
    return tf.math.confusion_matrix(gt_image, predicted_image)


def drd_fn(gt_image, predicted_image):
    rows, cols = np.shape(gt_image)
    n = 2
    m = 2*n+1
    m_range = np.arange(m)
    xx, yy = np.meshgrid(m_range, m_range)
    c = m//2
    xx -= c
    yy -= c
    denominator = np.sqrt(xx*xx+yy*yy)
    denominator[c, c] = 1  # to avoid division by zero
    Wm = 1/denominator
    Wm[c, c] = 0
    Wnm = Wm/np.sum(Wm)

    diff_index = np.nonzero(gt_image != predicted_image)
    DRD = 0
    for x,y in zip(diff_index[0], diff_index[1]):
        gk = predicted_image[x, y]
        '''
        u = np.max([0, x-(m//2)])
        b = np.min([rows-1, x+m//2])
        l = np.max([0, y-(m//2)])
        r = np.min([cols-1, y+m//2])
        '''
        Bk = np.zeros([m,m], dtype=int)
        for i in range(5):
            for j in range(5):
                ii = i-m//2
                jj = j-m//2
                if x+ii<0 or y+jj<0 or x+ii>=rows or y+jj>=cols:
                    Bk[i,j] = gk
                else:
                    Bk[i,j] = gt_image[x+ii,y+jj]
        Dk = np.abs(Bk-gk)
        DRDk = np.sum(np.multiply(Wnm,Dk))
        DRD += DRDk

    blocks = view_as_windows(gt_image, window_shape=(8, 8), step=8)
    blocks = blocks.reshape(blocks.shape[0]*blocks.shape[1], 8, 8)
    NUBN = 0
    for b in blocks:
        block_sum = np.sum(b)
        if block_sum != 0 and block_sum != b.size:
            NUBN += 1
    DRD /= NUBN
    return DRD


def nrm(y_true, y_pred):
    nrm_tf = losses.NRMLoss()
    return nrm_tf(y_true, y_pred).numpy()


def mpm(gt_image, predicted_image):
    fp = np.zeros(gt_image.shape)
    fp[(predicted_image == 0) & (gt_image == 1)] = 1
    fn = np.zeros(gt_image.shape)
    fn[(predicted_image == 1) & (gt_image == 0)] = 1

    kernel = np.ones((3, 3), dtype=np.uint8)
    im_dil = cv2.erode(gt_image, kernel)
    im_gtb = gt_image - im_dil
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
    return (mpfp + mpfn) / 2


def f1_score(gt_image, predicted_image, cfm=None):
    if cfm is None:
        cfm = confusion_matrix(gt_image, predicted_image)

    tp = cfm[0, 0]
    fp = cfm[1, 0]
    fn = cfm[0, 1]

    rec_values = tp / (tp + fn)
    prec_values = tp / (tp + fp)

    print('Precision: {}'.format(prec_values))
    print('Recall: {}'.format(rec_values))

    return (2 * rec_values * prec_values) / (rec_values + prec_values), prec_values, rec_values


class FMMetric(tf.keras.metrics.Metric):
    def __init__(self, name='binary_fm', **kwargs):
        super(FMMetric, self).__init__(name=name, **kwargs)
        self.fm = self.add_weight(name='fm', initializer='zeros')

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

        print('Precision: {}'.format(prec_values))
        print('Recall: {}'.format(rec_values))

        fm_values = (2 * rec_values * prec_values) / (rec_values + prec_values)

        self.fm.assign(fm_values)

    def result(self):
        return self.fm


class PSNRMetric(tf.keras.metrics.Metric):
    def __init__(self, name='binary_psnr', **kwargs):
        super(PSNRMetric, self).__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name='psnr', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        func = losses.InvPSNRLoss()
        self.psnr.assign(1.0/func(y_true, y_pred))

    def result(self):
        return self.psnr


class InvPSNRMetric(tf.keras.metrics.Metric):
    def __init__(self, name='binary_inv_psnr', **kwargs):
        super(InvPSNRMetric, self).__init__(name=name, **kwargs)
        self.inv_psnr = self.add_weight(name='inv_psnr', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        func = losses.InvPSNRLoss()
        self.inv_psnr.assign(func(y_true, y_pred))

    def result(self):
        return self.inv_psnr


class DiceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='binary_dice', **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.dice = self.add_weight(name='dice', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        func = losses.DiceLoss()
        self.dice.assign(func(y_true, y_pred))

    def result(self):
        return self.dice


class MicroFMMetric(tf.keras.metrics.Metric):
    def __init__(self, name='micro_fm', **kwargs):
        super(MicroFMMetric, self).__init__(name=name, **kwargs)
        self.micro_fm = self.add_weight(name='micro_fm', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        func = losses.MicroFMLoss()
        self.micro_fm.assign(func(y_true, y_pred))

    def result(self):
        return self.micro_fm



