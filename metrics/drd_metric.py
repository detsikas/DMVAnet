import tensorflow as tf
import numpy as np
from skimage.util.shape import view_as_windows


class DRDMetric():

    def __init__(self, name='DRD', **kwargs):
        self.drd = 0
        self.count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        rows, cols = np.shape(y_pred)
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

        diff_index = np.nonzero(y_true != y_pred)
        values = 0
        for x, y in zip(diff_index[0], diff_index[1]):
            gk = y_pred[x, y]
            '''
            u = np.max([0, x-(m//2)])
            b = np.min([rows-1, x+m//2])
            l = np.max([0, y-(m//2)])
            r = np.min([cols-1, y+m//2])
            '''
            Bk = np.zeros([m, m], dtype=int)
            for i in range(5):
                for j in range(5):
                    ii = i-m//2
                    jj = j-m//2
                    if x+ii < 0 or y+jj < 0 or x+ii >= rows or y+jj >= cols:
                        Bk[i, j] = gk
                    else:
                        Bk[i, j] = y_true[x+ii, y+jj]
            Dk = np.abs(Bk-gk)
            DRDk = np.sum(np.multiply(Wnm, Dk))
            values += DRDk

        blocks = view_as_windows(y_true, window_shape=(8, 8), step=8)
        blocks = blocks.reshape(blocks.shape[0]*blocks.shape[1], 8, 8)
        NUBN = 0
        for b in blocks:
            block_sum = np.sum(b)
            if block_sum != 0 and block_sum != b.size:
                NUBN += 1
        values /= NUBN

        self.drd += values
        self.count += 1

    def result(self):
        return self.drd/self.count


class DRDMetric_tf(tf.keras.metrics.Metric):

    def __init__(self, name='DRD', **kwargs):
        super(DRDMetric, self).__init__(name=name, **kwargs)
        self.drd = self.add_weight(name='drd', initializer='zeros')
        self.count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        print(y_true.dtype)
        print(y_pred.dtype)
        rows, cols = tf.shape(y_true)
        n = 2
        m = 2*n+1
        m_range = tf.range(m)
        xx, yy = tf.meshgrid(m_range, m_range)
        c = m//2
        xx -= c
        yy -= c
        denominator = tf.Variable(np.sqrt(xx*xx+yy*yy))
        denominator[c, c].assign(1)  # to avoid division by zero
        Wm = tf.Variable(1/denominator)
        Wm[c, c].assign(0)
        Wnm = Wm/tf.reduce_sum(Wm)

        diff_index = np.nonzero(y_true != y_pred)
        print(diff_index[0].shape)
        values = 0
        kkkkk = 0
        for x, y in zip(diff_index[0], diff_index[1]):
            print(kkkkk)
            kkkkk += 1
            gk = y_pred[x, y]
            '''
            u = np.max([0, x-(m//2)])
            b = np.min([rows-1, x+m//2])
            l = np.max([0, y-(m//2)])
            r = np.min([cols-1, y+m//2])
            '''
            Bk = tf.Variable(tf.zeros([m, m], dtype=tf.int64))
            for i in range(5):
                for j in range(5):
                    ii = i-m//2
                    jj = j-m//2
                    if x+ii < 0 or y+jj < 0 or x+ii >= rows or y+jj >= cols:
                        Bk[i, j].assign(gk)
                    else:
                        Bk[i, j].assign(y_true[x+ii, y+jj])
            Dk = tf.math.abs(Bk-gk)
            DRDk = tf.reduce_sum(tf.math.multiply(
                tf.cast(Wnm, tf.float32), tf.cast(Dk, tf.float32)))
            values += tf.cast(DRDk, tf.int64)

        blocks = view_as_windows(y_true, window_shape=(8, 8), step=8)
        blocks = blocks.reshape(blocks.shape[0]*blocks.shape[1], 8, 8)
        NUBN = 0
        for b in blocks:
            block_sum = np.sum(b)
            if block_sum != 0 and block_sum != b.size:
                NUBN += 1
        values /= NUBN

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        self.drd.assign_add(values)
        self.count += 1

    def result(self):
        return self.drd/self.count
