
import os
import sys
import tensorflow as tf
import cv2
import argparse
import numpy as np
import random
import facenet
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, BatchNormalization, Lambda, Input
from keras.models import Sequential, Model
from keras import backend as K

def load_dataset(path, validation_split):
    '''
    Args: path => Path of train directory or test directory
    Returns:
    - X_train: training images in nparray
    - Y_train: training labels
    - num_examples: a list mapping each class to their number of examples.
    - num_classes
    '''
    X_train, Y_train  = [], []
    num_classes = 0
    classes = sorted([int(i) for i in os.listdir(path) if i.isdigit()])
    for cls in classes:
        num_classes += 1
        class_path = os.path.join(path, str(cls))
        # If not a directory -> skip
        if not os.path.isdir(class_path):
             continue
        images = os.listdir(class_path)

        for image in images:
            image_path = os.path.join(class_path, image)
            img = cv2.imread(image_path)
            X_train.append(img)
            Y_train.append(cls)

        if num_classes % 200 == 0:
            print("Loading class", cls)

    X_train = np.stack(X_train)
    Y_train = np.stack(Y_train)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_split)

    # By now, num_classes = total number of classes in the original dataset.
    num_examples = []

    unique, counts = np.unique(Y_train, return_counts=True)
    di = dict(zip(unique, counts))

    for i in range(num_classes):
        if i in di: num_examples.append(di[i])
        else: num_examples.append(0)
    # num_classes = total number of classes in training set AFTER splitting.
    num_classes = unique.shape[0]
    return X_train, Y_train, X_val, Y_val, num_classes, num_examples


def get_batch_train(batch_size, X_train, Y_train, num_classes, num_examples):
    """
    Create batch of n pairs, half from same class, half from different classes
    """

    _, w, h, __ = X_train.shape
    # Randomly sample several classes to use in the batch
    categories = np.random.choice(num_classes, size=(batch_size,), replace=False)
    # Initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, h, w)) for i in range(2)]
    labels = np.zeros((batch_size,))
    # Make one half '0's, and the another half '1's.
    labels[batch_size // 2:] = 1

    for i in range(batch_size):
        category = categories[i]
        # Get the number of examples of this 'category'.
        examples = num_examples[category]

        while i >= batch_size // 2 and examples == 1:
            category = np.random.randint(0, num_classes)
            examples = num_examples[category]

        i_1 = np.random.randint(0, examples)
        # Get the location (index) of that example in X_train and Y_train
        idx_1 = i_1 + sum(num_examples[:category])
        pairs[0][i, :, :] = X_train[idx_1, :, :]

        idx_2 = np.random.randint(0, num_classes)

        # Pick images of different class for the 1st half, same for the 2nd half.
        if i >= batch_size // 2:
            # Add a random number to the i_1 modulo numer of examples to ensure that two images are different.
            i_2 = (i_1 + np.random.randint(1, examples)) % examples
            idx_2 = i_2 + sum(num_examples[:category])
        else:
            # Add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + np.random.randint(1, num_classes)) % num_classes
            examples_2 = num_examples[category_2]
            i_2 = np.random.randint(0, examples_2)
            idx_2 = i_2 + sum(num_examples[:category_2])

        pairs[1][i, :, :] = X_train[idx_2, :, :]

    return pairs, labels



def get_batch_val(batch_size, X_val, Y_val):
    """
    Create batch of n pairs
    """

    num_imgs, w, h, _ = X_val.shape
    # Initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, h, w)) for i in range(2)]
    labels = np.zeros((batch_size,))

    for i in range(batch_size):

        idx_1 = np.random.randint(0, num_imgs)
        pairs[0][i, :, :] = X_val[idx_1, :, :]

        idx_2 = np.random.randint(0, num_imgs)
        while idx_1 == idx_2: idx_2 = np.random.randint(0, num_imgs)
        pairs[1][i, :, :] = X_val[idx_2, :, :]
        if Y_val[idx_1] == Y_val[idx_2]:
            labels[i] = 1
        else: labels[i] = 0

    return pairs, labels




def train_generator(tensors, batch_size, X_train, Y_train, num_classes, num_examples):
    sess, images_placeholder, phase_train_placeholder, embeddings = tensors
    while True:
        pairs, labels = get_batch_train(batch_size, X_train, Y_train, num_classes, num_examples)
        emb_array = []
        for i in range(2):
            feed_dict = {images_placeholder: pairs[i], phase_train_placeholder: False}
            emb_array.append(sess.run(embeddings, feed_dict=feed_dict))
        yield (emb_array, labels)


def val_generator(tensors, X_val, Y_val):
    sess, images_placeholder, phase_train_placeholder, embeddings = tensors
    while True:
        pairs, labels = get_batch_val(X_val.shape[0], X_val, Y_val)
        emb_array = []
        for i in range(2):
            feed_dict = {images_placeholder: pairs[i], phase_train_placeholder: False}
            emb_array.append(sess.run(embeddings, feed_dict=feed_dict))
        yield (emb_array, labels)



def main(args):

    np.random.seed(seed=args.seed)

    X_train, Y_train, X_val, Y_val, num_classes, num_examples = load_dataset(args.input_dir, args.validation_split)


    with tf.Graph().as_default():
        with tf.Session() as sess:

            # Convert X_val into emnedding features.
            facenet.load_model(args.model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            tensors = (sess, images_placeholder, phase_train_placeholder, embeddings)

            epochs = args.epochs
            batch_size = args.batch_size

            print('Number of classes: %d' % num_classes)
            print('Number of images: %d' % sum(num_examples))

            '''======================================================'''
            input_shape = (None, embedding_size)
            left_input = Input(input_shape)
            right_input = Input(input_shape)
            # Add a customized layer to compute the absolute difference between the encodings
            L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
            L1_distance = L1_layer([left_input, right_input])

            # Add a dense layer with a sigmoid unit to generate the similarity score
            prediction = Dense(1, activation='sigmoid')(L1_distance)

            model = Model(inputs=[left_input, right_input], outputs=prediction)

            '''======================================================'''

            model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(train_generator(tensors, batch_size, X_train, Y_train, num_classes, num_examples),
                      steps_per_epoch=epochs // batch_size, epochs=epochs,
                      validation_data=val_generator(tensors, X_val, Y_val), validation_steps=1)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with aligned training images.')
    parser.add_argument('model', type=str,
        help='''Pre-trained model for embedding generation. Could be either
        a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file''')
    parser.add_argument('validation_split', type=float,
        help='The fraction of validaton test set.')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('epochs', type=int)
    parser.add_argument('--image_size', type=int,
        help='Size (height, width) in pixels of the training images.', default=160)
    # parser.add_argument('--embedding_size', type=int,
    #     help='Length of the embedding vector.', default=512)
    parser.add_argument('--seed', type=int, default=4)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
