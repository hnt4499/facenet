"""Performs face alignment and stores face thumbnails in the output directory.
Modified from the file align_dataset_mtcnn.py by David Sandberg.
Use dlib to detect face(s).
Author: Hoang Nghia Tuyen."""

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

from scipy import misc
import sys
import os
import argparse
import dlib
import openface
import tensorflow as tf
import numpy as np
import facenet
import random
from time import sleep

def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)

    print('Loading predictor model')

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(args.model)
    face_aligner = openface.AlignDlib(args.model)


    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)

    nrof_images_total = 0
    nrof_successfully_aligned = 0
    nrof_unchanged_images = 0
    print('Total number of classes:', len(dataset))
    print('Number of classes to align:', args.num_classes)
    for cls in dataset[:args.num_classes]:
        print('Processing class:', cls.name)
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename+'.png')
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim<2:
                        print('Unable to align "%s"' % image_path)
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = img[:,:,0:3]

                    # Run the HOG face detector on the image data
                    detected_face = face_detector(img, 1)

                    if len(detected_face) > 0:
                        # As face(s) detected is/ are stored in a list, we need to pull it out.
                        face_rect = detected_face[0]
                    	# Get the the face's pose
                        pose_landmarks = face_pose_predictor(img, face_rect)
                    	# Use openface to calculate and perform the face alignment
                        alignedFace = face_aligner.align(534, img, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                        misc.imsave(output_filename, alignedFace)
                        nrof_successfully_aligned += 1

                    else:
                        filename_base, file_extension = os.path.splitext(output_filename)
                        # Add 'fail' to indicates that dlib cannot detect face in this image.
                        output_filename = "{}{}".format(filename_base + '_failed', file_extension)
                        nrof_unchanged_images += 1
                        misc.imsave(output_filename, img)

    print('Total number of images:', nrof_images_total)
    print('Number of successfully aligned images:', nrof_successfully_aligned)
    print('Number of resized images (MTCNN cannot detect face in these images):', nrof_unchanged_images)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('model', type=str,
        help='Pre-trained model for facial keypoints detection, named in the form shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--image_size', type=int,
        help='Size (height, width) in pixels of the destination images.', default=182)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--num_classes', type=int,
                        help='Number of classes to align using MTCNN. Useful for testing.', default=1)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
