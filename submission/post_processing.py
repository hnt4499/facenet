import os
import pandas as pd
import re
import sys
import argparse

def main(args):
    paths = [os.path.join(args.submission_path, i) for i in os.listdir(args.submission_path)]
    dest = args.output_dir
    submission = pd.DataFrame()
    for i in range(len(paths)):
        s = pd.read_csv(paths[i])
        submission = pd.concat([submission, s], ignore_index=True)
    submission.label = submission.label.apply(lambda x: re.sub(' +', ' ', x))
    submission.to_csv(dest, index=False)

def transform(x):
    x = re.sub(' +', ' ', x)
    x = x.lstrip()
    x = x.rstrip()
    return x


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('submission_path', type=str, help='Path to the submissions, not including the names themselves')
    parser.add_argument('output_dir', type=str, help='Output directory of the submission.csv file')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
