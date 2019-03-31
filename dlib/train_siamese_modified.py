
import os
import sys
import tensorflow as tf
import cv2
import argparse
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, BatchNormalization, Lambda, Input
from keras.models import Sequential, Model
from keras import backend as K

def load_dataset(X_path, Y_path, validation_split, tot_nrof_classes):
    '''
    Args: paths: paths of extracted features.
    Returns:
    - X: training/validation images in nparray
    - Y: training/validaton labels
    - nrof_examples: a list mapping each class to their number of examples.
    - nrof_classes: total number of classes in training set AFTER splitting.
    '''
    X_train = np.load(X_path)
    Y_train = np.load(Y_path)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_split)

    nrof_examples = []

    unique, counts = np.unique(Y_train, return_counts=True)
    di = dict(zip(unique, counts))

    for i in range(tot_nrof_classes):
        if i in di: nrof_examples.append(di[i])
        else: nrof_examples.append(0)

    # nrof_classes = total number of classes in training set AFTER splitting.
    nrof_classes = unique.shape[0]

    print('X_train shape: {}, Y_train shape: {}'.format(X_train.shape, Y_train.shape))
    print('X_val shape: {}, Y_val shape: {}'.format(X_val.shape, Y_val.shape))
    return X_train, Y_train, X_val, Y_val, nrof_classes, nrof_examples


def get_batch_train(batch_size, X_train, Y_train, nrof_classes, nrof_examples, embedding_size):
    """
    Create batch of n pairs, half from same class, half from different classes
    """

    # Randomly sample several classes to use in the batch
    categories = np.random.choice(nrof_classes, size=(batch_size,), replace=False)
    # Initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, embedding_size)) for i in range(2)]
    labels = np.zeros((batch_size,))
    # Make one half '0's, and the another half '1's.
    labels[batch_size // 2:] = 1

    for i in range(batch_size):
        category = categories[i]
        # Get the number of examples of this 'category'.
        examples = nrof_examples[category]

        # Check if the selected class has only 1 or 0 training example.
        while (i >= batch_size // 2 and examples == 1) or examples == 0:
            category = np.random.randint(0, nrof_classes)
            examples = nrof_examples[category]

        i_1 = np.random.randint(0, examples)
        # Get the location (index) of that example in X_train and Y_train
        idx_1 = i_1 + sum(nrof_examples[:category])
        pairs[0][i, :] = X_train[idx_1, :]

        # Pick images of different class for the 1st half, same for the 2nd half.
        if i >= batch_size // 2:
            # Add a random number to the i_1 modulo numer of examples to ensure that two images are different.
            i_2 = (i_1 + np.random.randint(1, examples)) % examples
            idx_2 = i_2 + sum(nrof_examples[:category])
        else:
            # Add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + np.random.randint(1, nrof_classes)) % nrof_classes
            examples_2 = nrof_examples[category_2]

            # Check if the selected class has only 1 or 0 training example.
            while examples_2 == 0:
                category_2 = np.random.randint(0, nrof_classes)
                examples_2 = nrof_examples[category_2]

            i_2 = np.random.randint(0, examples_2)
            idx_2 = i_2 + sum(nrof_examples[:category_2])

        pairs[1][i, :] = X_train[idx_2, :]

    return pairs, labels



def get_batch_val(batch_size, X_val, Y_val, embedding_size):
    """
    Create batch of n pairs
    """

    # Initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, embedding_size)) for i in range(2)]
    labels = np.zeros((batch_size,))

    nrof_imgs = X_val.shape[0]
    for i in range(batch_size):

        idx_1 = np.random.randint(0, nrof_imgs)
        pairs[0][i, :] = X_val[idx_1, :]

        idx_2 = np.random.randint(0, nrof_imgs)
        while idx_1 == idx_2: idx_2 = np.random.randint(0, nrof_imgs)

        pairs[1][i, :] = X_val[idx_2, :]

        if Y_val[idx_1] == Y_val[idx_2]:
            labels[i] = 1
        else: labels[i] = 0

    return pairs, labels




def train_generator(batch_size, X_train, Y_train, nrof_classes, nrof_examples, embedding_size):
    while True:
        pairs, labels = get_batch_train(batch_size, X_train, Y_train, nrof_classes, nrof_examples, embedding_size)
        yield (pairs, labels)


def val_generator(X_val, Y_val, embedding_size):
    while True:
        pairs, labels = get_batch_val(X_val.shape[0], X_val, Y_val, embedding_size)
        yield (pairs, labels)



def main(args):

    np.random.seed(seed=args.seed)

    X_path = os.path.join(args.features_dir, args.embeddings_name)
    Y_path = os.path.join(args.features_dir, args.labels_name)

    X_train, Y_train, X_val, Y_val, nrof_classes, nrof_examples = load_dataset(X_path, Y_path, args.validation_split, args.tot_nrof_classes)

    epochs = args.epochs
    batch_size = args.batch_size
    embedding_size = args.embedding_size

    print('Number of classes: %d' % nrof_classes)
    print('Number of images: %d' % sum(nrof_examples))

    '''======================================================'''
    input_shape = (embedding_size, )
    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([left_input, right_input])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)

    model = Model(inputs=[left_input, right_input], outputs=prediction)

    model.summary()

    '''======================================================'''

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_generator(batch_size, X_train, Y_train,
                                        nrof_classes, nrof_examples, embedding_size),
                        steps_per_epoch=epochs // batch_size, epochs=epochs, verbose=0,
                        validation_data=val_generator(X_val, Y_val, embedding_size), validation_steps=1)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('features_dir', type=str, help='Directory of extracted features of training set')
    parser.add_argument('--embeddings_name', type=str,
        help='String of which the embeddings numpy array is saved as.',
        default='embeddings.npy')
    parser.add_argument('--labels_name', type=str,
        help='String of which the labels numpy array is saved as. This assumes that all classes are of integer.',
        default='labels.npy')

    parser.add_argument('validation_split', type=float,
        help='The fraction of validaton test set.')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('epochs', type=int)
    parser.add_argument('--image_size', type=int,
        help='Size (height, width) in pixels of the training images.', default=160)
    parser.add_argument('--tot_nrof_classes', type=int,
        help='Total number of classes in the original traing set.', default=1000)
    parser.add_argument('--embedding_size', type=int,
        help='Length of the embedding vector.', default=512)
    parser.add_argument('--seed', type=int, default=4)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
