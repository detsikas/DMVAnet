import argparse
import sys
import os
import tensorflow as tf
import common.model_info_io as model_info_io

parser = argparse.ArgumentParser(description='Create a summary from a saved model')
parser.add_argument('model_path', help='Path to model directory')
args = parser.parse_args()

model_path = args.model_path

if not os.path.exists(model_path) or not os.path.isdir(model_path):
    print('Bad model path')
    sys.exit(0)

model_info = model_info_io.read_info(os.path.join(model_path, 'info'))
print(model_info.type)
model = model_info_io.restore_model(model_info)
model.load_weights(os.path.join(model_path, 'weights'))
model.summary()
