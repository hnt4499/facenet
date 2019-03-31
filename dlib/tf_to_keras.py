#!/usr/bin/env python
# coding: utf-8

# # TensorFlow to Keras

# ### Reference
# - https://github.com/myutwo150/keras-inception-resnet-v2

# ### Pre-trained tensorflow models
#
# - https://github.com/davidsandberg/facenet

import sys
import os
import argparse
import re
import numpy as np
import tensorflow as tf
from inception_resnet_v1 import *
from keras.layers import Input


def get_filename(key):
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('InceptionResnetV1_', '')

    # remove "Repeat" scope from filename
    filename = re_repeat.sub('B', filename)

    if re_block8.match(filename):
        # the last block8 has different name with the previous 5 occurrences
        filename = filename.replace('Block8', 'Block8_6')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def extract_tensors_from_checkpoint_file(filename, output_folder):
    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # not saving the following tensors
        if key == 'global_step':
            continue
        if 'AuxLogit' in key:
            continue

        # convert tensor name into the corresponding Keras layer weight name and save
        path = os.path.join(output_folder, get_filename(key))
        arr = reader.get_tensor(key)
        np.save(path, arr)

def main(args):
    npy_weights_dir = os.path.join(args.output_dir, 'npy_weights')
    weights_dir = os.path.join(args.output_dir, 'weights')
    model_dir = os.path.join(args.output_dir, 'model')

    weights_filename = 'facenet_keras_weights.h5'
    model_filename = 'facenet_keras.h5'

    os.makedirs(npy_weights_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # regex for renaming the tensors to their corresponding Keras counterpart
    re_repeat = re.compile(r'Repeat_[0-9_]*b')
    re_block8 = re.compile(r'Block8_[A-Za-z]')

    extract_tensors_from_checkpoint_file(args.input_dir, npy_weights_dir)

    inputs = tf.placeholder(dtype='float', shape=[None, args.image_size, args.image_size, 3])
    model, _ = InceptionResNetV1(inputs=inputs)


    print('Loading numpy weights from', npy_weights_dir)
    for layer in model.layers:
        if layer.weights:
            weights = []
            for w in layer.weights:
                weight_name = os.path.basename(w.name).replace(':0', '')
                weight_file = layer.name + '_' + weight_name + '.npy'
                weight_arr = np.load(os.path.join(npy_weights_dir, weight_file))
                weights.append(weight_arr)
            layer.set_weights(weights)

    print('Saving weights...')
    model.save_weights(os.path.join(weights_dir, weights_filename))
    print('Saving model...')
    model.save(os.path.join(model_dir, model_filename))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='Directory pointing to pretrained model, i.e. ckpt-250000 file.')
    parser.add_argument('output_dir', type=str, help='Output directory.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
