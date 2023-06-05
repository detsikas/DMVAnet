import tensorflow as tf

class PSNRMetric(tf.keras.metrics.Metric):
    def __init__(self, name='psnr', **kwargs):
        super(PSNRMetric, self).__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name='psnr', initializer='zeros')
        self.count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        value = tf.image.psnr(y_true, y_pred, max_value=1)
        self.psnr.assign_add(value)
        self.count+=1

  def result(self):
    return self.psnr/self.count