import tensorflow as tf
import numpy as np
from models.models import build_model
import datasets.dataset_utils as dataset_utils
import argparse
import yaml
import os
import sys
from metrics.pf1_metric import PseudoF1Metric
from metrics.drd_metric import DRDMetric
from metrics.nrm_metric import NRMMetric
from metrics.mpm_metric import MPMMetric
from metrics.psnr_metric import PSNRMetric
from metrics.dice_metric import DiceMetric
from metrics.f1_metric import F1Metric

parser = argparse.ArgumentParser(
    description='Predict validation images')
parser.add_argument(
    '--model-path', help='Path to trained model', required=True)
parser.add_argument('--stride', help='Image patch stride',
                    required=True, type=int)
parser.add_argument(
    '--multiscale', help='Multiscale testing', action='store_true')
parser.add_argument(
    '--subset', help='Dataset subset path to use for validation')
args = parser.parse_args()

model_path = args.model_path
stride = args.stride
multiscale = args.multiscale
subset = args.subset

if not os.path.exists(model_path) or not os.path.isdir(model_path):
    print('Bad model path')
    sys.exit(0)

config_file = os.path.join(model_path, 'config.yaml')

if not os.path.exists(config_file) or not os.path.isfile(config_file):
    print('Config file missing')
    sys.exit(0)

with open(config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

target_image_size = config['target_image_size']
validation_dataset_path = config['validation_dataset_path']

if subset is not None:
    validation_dataset_path = os.path.join(validation_dataset_path, subset)

if not os.path.exists(validation_dataset_path) or not os.path.isdir(validation_dataset_path):
    print('Bad validation path')
    sys.exit(0)

model_type = config['model_type']
batch_size = 1
input_shape = [target_image_size, target_image_size, 3]

print('Input shape: {}'.format(input_shape))
print('Model type: {}'.format(model_type))

# Restore the model
model = build_model(model_type=model_type, model_shape=input_shape)
model.load_weights(os.path.join(model_path, 'model', 'weights'))

# Read the validation dataset
validation_dataset = dataset_utils.create_dataset_inference_pipeline(
    source_directory=validation_dataset_path)


def reconstruct_image(predictions_, h_anchors, w_anchors, whole_image_shape):
    target_image_size_ = predictions_.shape[1]
    number_of_classes_ = predictions_.shape[-1]
    predictions_ = tf.reshape(predictions_, [h_anchors.shape[0],
                                             w_anchors.shape[0], target_image_size_, target_image_size_, number_of_classes_])
    merged = np.zeros(
        [whole_image_shape[0], whole_image_shape[1], predictions_.shape[-1]])
    count = np.zeros_like(merged)

    for i, h in enumerate(h_anchors):
        for j, w in enumerate(w_anchors):
            merged[h:h+target_image_size_, w:w +
                   target_image_size_] += predictions_[i, j]
            count[h:h+target_image_size_, w:w+target_image_size_] += 1

    merged = merged/count

    return merged


if multiscale:
    img_ratios = [0.75, 1.0, 1.25, 1.5, 1.75]

else:
    img_ratios = [1.0]


mse_metric = [tf.keras.metrics.MeanSquaredError()
              for i in range(len(img_ratios))]
binary_cross_entropy_metric = [
    tf.keras.metrics.BinaryCrossentropy() for i in range(len(img_ratios))]
binary_accuracy_metric = [tf.keras.metrics.BinaryAccuracy()
                          for i in range(len(img_ratios))]
precision_metric = [tf.keras.metrics.Precision()
                    for i in range(len(img_ratios))]
recall_metric = [tf.keras.metrics.Recall() for i in range(len(img_ratios))]
f1_metric = np.zeros(len(img_ratios))
pf1_metric = [PseudoF1Metric() for i in range(len(img_ratios))]
drd_metric = [DRDMetric() for i in range(len(img_ratios))]
nrm_metric = [NRMMetric() for i in range(len(img_ratios))]
mpm_metric = [MPMMetric() for i in range(len(img_ratios))]  # *1000
psnr_metric = [PSNRMetric() for i in range(len(img_ratios))]  # threshold image
dice_metric = [DiceMetric() for i in range(len(img_ratios))]


def threshold_image(img):
    local_image = np.copy(img)
    local_image[local_image <= 0.5] = 0
    local_image[local_image > 0.5] = 1
    return local_image.astype('uint8')


for i, (x, y) in enumerate(validation_dataset):
    for j, ratio in enumerate(img_ratios):
        image_size = tf.shape(x)
        patches, h_anchors, w_anchors = dataset_utils.extract_inference_patches(
            x, target_image_size, stride)
        predictions = model.predict(patches, verbose=0)
        reconstructed_image = reconstruct_image(
            predictions, h_anchors, w_anchors, [image_size[0], image_size[1], x.shape[-1]])

        mse_metric[j].update_state(y, reconstructed_image)
        binary_cross_entropy_metric[j].update_state(y, reconstructed_image)
        binary_accuracy_metric[j].update_state(y, reconstructed_image)
        precision_metric[j].update_state(
            y.numpy().flatten(), reconstructed_image.flatten())
        recall_metric[j].update_state(
            y.numpy().flatten(), reconstructed_image.flatten())
        f1_metric[j] = 2.0*precision_metric[j].result().numpy()*recall_metric[j].result().numpy() / \
            (precision_metric[j].result().numpy() +
             recall_metric[j].result().numpy())
        pf1_metric[j].update_state(tf.squeeze(
            y, axis=-1).numpy(), tf.squeeze(reconstructed_image, axis=-1).numpy())
        drd_metric[j].update_state(tf.squeeze(
            y, axis=-1).numpy().astype(int), threshold_image(tf.squeeze(reconstructed_image, axis=-1).numpy()).astype(int))
        # nrm_metric[j].update_state(y, reconstructed_image)
        # pm_metric[j].update_state(y, reconstructed_image)
        psnr_metric[j].update_state(y, reconstructed_image)
        # dice_metric[j].update_state(y, reconstructed_image)

    values = [mse_metric[j].result().numpy() for j in range(len(img_ratios))]
    print(
        f'Batch {i} MSE :\t{values} - mean {np.mean(values)}')

    values = [binary_cross_entropy_metric[j].result().numpy()
              for j in range(len(img_ratios))]
    print(
        f'Batch {i} BCE:\t{values} - mean {np.mean(values)}')

    values = [binary_accuracy_metric[j].result().numpy()
              for j in range(len(img_ratios))]
    print(
        f'Batch {i} BAC:\t{values} - mean {np.mean(values)}')

    values = [precision_metric[j].result().numpy()
              for j in range(len(img_ratios))]
    print(
        f'Batch {i} PREC:\t{values} - mean {np.mean(values)}')

    values = [recall_metric[j].result().numpy()
              for j in range(len(img_ratios))]
    print(
        f'Batch {i} REC:\t{values} - mean {np.mean(values)}')

    values = [f1_metric[j]
              for j in range(len(img_ratios))]
    print(
        f'Batch {i} f1:\t{values} - mean {np.mean(values)}')

    values = [pf1_metric[j].result()['p_f1'].numpy()
              for j in range(len(img_ratios))]
    print(
        f'Batch {i} pf1:\t{values} - mean {np.mean(values)}')

    values = [drd_metric[j].result()
              for j in range(len(img_ratios))]
    print(
        f'Batch {i} drd:\t{values} - mean {np.mean(values)}')

    values = [psnr_metric[j].result().numpy()
              for j in range(len(img_ratios))]
    print(
        f'Batch {i} PSNR:\t{values} - mean {np.mean(values)}')
    print('---------------')


print('Done')
