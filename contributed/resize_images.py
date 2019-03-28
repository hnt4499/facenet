from keras.preprocessing.image import load_img, save_img
import os
import sys
import argparse

def main(args):
    images = os.listdir(args.input_dir)
    count = 1
    for img in images:
        if count % 1000 == 0:
            print('Processing image ', count)

        src = os.path.join(args.input_dir, img)
        dest = os.path.join(args.output_dir, img)
        save_img(dest, load_img(src).resize((args.image_size, args.image_size)))
        count += 1

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with original images.')
    parser.add_argument('output_dir', type=str, help='Directory with resized images.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
