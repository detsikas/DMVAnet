import numpy as np
import random


def gaussian_noise(image, mean=0, var=0.001):
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    out = np.clip(out, 0, 1.0)
    return out.astype('float32')


def sp_noise(image, prob):
    output = np.zeros(image.shape, image.dtype)
    threshold = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > threshold:
                output[i][j] = 1
            else:
                output[i][j] = image[i][j]
    return output
