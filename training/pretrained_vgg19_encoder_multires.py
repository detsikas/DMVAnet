import tensorflow as tf
import argparse
from common.dataset_utils import get_dataset
import os
import datetime
import common.models as models
import common.model_info_io as model_info_io
import common.metrics as metrics
import common.losses as losses

parser = argparse.ArgumentParser(description='Train binarization model')
training_group = parser.add_mutually_exclusive_group(required=True)
training_group.add_argument('--training-dataset-files', nargs='+', help='List of tf record files with training datasets')
training_group.add_argument('--training-dataset-dir', help='Folder with tf record files with training datasets')
testing_group = parser.add_mutually_exclusive_group(required=True)
testing_group.add_argument('--testing-dataset-files', nargs='+', help='List of tf record files with testing datasets')
testing_group.add_argument('--testing-dataset-dir', help='Folder with tf record files with testing datasets')
parser.add_argument('--output-dir', help='Output directory (where everything goes)')
parser.add_argument('--model-only', help='Output model only, without training', action='store_true')
parser.add_argument('--epochs', help='Number of raining epochs', default=20, type=int)

args = parser.parse_args()
training_dataset_files = args.training_dataset_files
training_dataset_dir = args.training_dataset_dir
testing_dataset_files = args.testing_dataset_files
testing_dataset_dir = args.testing_dataset_dir
output_dir = args.output_dir
model_only = args.model_only
epochs = args.epochs

train_dataset = get_dataset(training_dataset_files, training_dataset_dir, 1000, 32)
test_dataset = get_dataset(testing_dataset_files, testing_dataset_dir, 1000, 32)

input_shape = [256, 256, 3]
model_info = model_info_io.Info(input_shape, model_info_io.ModelType.PRETRAINED_VGG19_ENCODER_MULTIRESNET_UNET)
model, pretrained_model = models.multires_unet_with_vgg19_pretrained_input(input_shape)

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam_optimizer, loss=losses.DiceLoss(),
              metrics=['binary_accuracy', 'mean_squared_error'])
print(model.summary())
#tf.keras.utils.plot_model(model, "res_unet.png", show_shapes=True)

if not model_only:
    model_dir = os.path.join(output_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint_dir = os.path.join(model_dir, 'weights')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_best_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'tensorboard_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), histogram_freq=1)
    history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset,
                        callbacks=[tensorboard_callback, model_checkpoint_callback])

    pretrained_model.trainable = True
    model.compile(optimizer='adam', loss=losses.DiceLoss(),
                              metrics=['binary_accuracy', 'mean_squared_error', metrics.InvPSNRMetric(),
                                                            metrics.NRMMetric()])#, metrics.FMMetric()])

    history_2 = model.fit(train_dataset, epochs=epochs+epochs//3, validation_data=test_dataset, initial_epoch=epochs,
                                            callbacks=[tensorboard_callback, model_checkpoint_callback])

    model_info_io.write_info(model_info, os.path.join(model_dir, 'info'))

