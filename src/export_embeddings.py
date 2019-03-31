"""
Exports the embeddings and labels of a directory of images as numpy arrays.

Output:
embeddings.npy -- Embeddings as np array, Use --embeddings_name to change name
labels.npy -- Integer labels as np array, Use --labels_name to change name
label_strings.npy -- Strings from folders names, --labels_strings_name to change name

Modified from the export_embedding.py file by Charles Jekel.

Use --image_batch to dictacte how many images to load in memory at a time.

Author: Hoang Nghia Tuyen

"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
import glob
import math
import cv2

from six.moves import xrange


def load_dataset(path, classes):
    '''
    Args:
    - path: path of train directory or test directory
    - classes: classes to be processed
    Returns:
    - X: training images in nparray
    - Y: training labels
    - nrof_images_batch: total number of images in this batch.
    '''
    X, Y  = [], []

    nrof_images_batch = 0
    for cls in classes:

        class_path = os.path.join(path, str(cls))
        # If not a directory -> skip
        if not os.path.isdir(class_path):
             continue
        images = os.listdir(class_path)

        for image in images:
            image_path = os.path.join(class_path, image)
            img = cv2.imread(image_path)

            X.append(img)
            Y.append(cls)

            nrof_images_batch += 1
            if nrof_images_batch % 500 == 1:
                print("Loading image", cls)


    X = np.stack(X)
    Y = np.stack(Y)

    return X, Y, nrof_images_batch



def main(args):
    input_dir = args.input_dir
    classes = sorted([int(i) for i in os.listdir(input_dir) if i.isdigit()])
    nrof_classes = len(classes)
    nrof_images = sum([len(os.listdir(os.path.join(input_dir, str(i)))) for i in classes])

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            print('Number of classes:', nrof_classes)
            print('Number of images:', nrof_images)

            # Note that batch_size indicates how many CLASSES are stored in memory at a time.
            batch_size = args.class_batch

            nrof_batches = math.ceil(nrof_classes / batch_size)

            print('Number of batches:', nrof_batches)

            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((nrof_images, embedding_size))
            labels = np.zeros((nrof_images,))

            start_time = time.time()

            start_image = 0

            for i in range(nrof_batches):
                if i == nrof_batches - 1:
                    n = nrof_classes
                else:
                    n = i * batch_size + batch_size
                # Get images for the batch
                X, Y, nrof_images_batch = load_dataset(input_dir, classes[i * batch_size:n])

                labels[start_image:start_image + nrof_images_batch] = Y

                feed_dict = {images_placeholder: X, phase_train_placeholder:False}

                # Use the facenet model to calcualte embeddings
                embed = sess.run(embeddings, feed_dict=feed_dict)
                emb_array[start_image:start_image + nrof_images_batch, :] = embed

                # Update the start index of the batch
                start_image += nrof_images_batch

                print('Completed batch', i+1, 'of', nrof_batches)

            run_time = time.time() - start_time
            print('Run time: ', run_time)

            # Export embeddings and labels
            labels  = np.stack(labels, axis=0)

            # Create a list that maps each class to their number of examples.
            unique, examples = np.unique(labels, return_counts=True)

            # Exporting...
            output_dir = args.output_dir
            # This is for compatibility of Kaggle kernel.
            if output_dir == '':
                emb_dir = args.embeddings_name
                labels_dir = args.labels_name
                examples_dir = args.examples_name
            else:
                emb_dir = os.path.join(output_dir, args.embeddings_name)
                labels_dir = os.path.join(output_dir, args.labels_name)
                examples_dir = os.path.join(output_dir, args.examples_name)

            np.save(emb_dir, emb_array)
            np.save(labels_dir, labels)
            np.save(examples_dir, examples)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
        help='Directory containing the meta_file and ckpt_file.')
    parser.add_argument('input_dir', type=str,
        help='Directory containing images.')
    parser.add_argument('--output_dir', type=str,
        help='Directory to which exported features will be saved.', default='')
    parser.add_argument('--class_batch', type=int,
        help='Number of class(es) stored in memory at a time.', default=2)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.',
        default=1.0)

    # Numpy file Names
    parser.add_argument('--embeddings_name', type=str,
        help='String of which the embeddings numpy array is saved as.',
        default='embeddings.npy')
    parser.add_argument('--labels_name', type=str,
        help='String of which the labels numpy array is saved as. This assumes that all classes are of integer.',
        default='labels.npy')
    parser.add_argument('--examples_name', type=str,
        help='String of which the nrof_examples numpy array is saved as.',
        default='examples.npy')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
