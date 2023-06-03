import tensorflow as tf
import argparse
import common.dataset_utils as dataset_utils
import os
import datetime
import common.model_info_io as model_info_io
import common.losses as losses
import yaml
import sys

parser = argparse.ArgumentParser(description='Train binarization model')
parser.add_argument('--config-file', help='Configuration file', required=True)
parser.add_argument(
    '--output-dir', help='Output directory (where everything goes)')
parser.add_argument(
    '--model-only', help='Output model only, without training', action='store_true')
parser.add_argument(
    '--epochs', help='Number of raining epochs', default=20, type=int)

args = parser.parse_args()
config_file = args.config_file

if not os.path.exists(config_file) or not os.path.isfile(config_file):
    print('Bad configuration file')
    sys.exit(0)

output_dir = args.output_dir
model_only = args.model_only

with open(config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

target_image_size = config['target_image_size']
batch_size = config['batch_size']
epochs = config['epochs']
training_dataset_path = config['training_dataset_path']
validation_dataset_path = config['validation_dataset_path']
model_type = config['model_type']

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'config.yaml'), "w") as f:
    yaml.dump(config, f)


train_dataset = dataset_utils.create_dataset_training_pipeline(source_directory=training_dataset_path,
                                                               batch_size=batch_size, target_size=target_image_size, augment=True)
validation_dataset = dataset_utils.create_dataset_training_pipeline(source_directory=validation_dataset_path,
                                                                    batch_size=batch_size, target_size=target_image_size, augment=False)

input_shape = [target_image_size, target_image_size, 3]
model_info = model_info_io.Info(input_shape, model_type)
model = model_info_io.restore_model(model_info)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam_optimizer, loss=losses.DiceLoss(),
              metrics=['binary_accuracy', 'mean_squared_error'])
print(model.summary())
# tf.keras.utils.plot_model(model, "base_unet.png", show_shapes=True)

if not model_only:
    model_dir = os.path.join(output_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint_dir = os.path.join(model_dir, 'weights')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir, save_best_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(output_dir, 'tensorboard_{}'.format(
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),
        histogram_freq=1)
    history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset,
                        callbacks=[tensorboard_callback, model_checkpoint_callback])

    model_info_io.write_info(model_info, os.path.join(model_dir, 'info'))
