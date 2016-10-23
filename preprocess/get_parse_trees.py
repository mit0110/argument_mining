"""Script to create parse trees and save them in pickle format.

Usage:
    lr_baseline.py --input_filename=<file> --output_filename=<file>

Options:
    --input_filename=<file>     The path to directory to read the dataset.
    --output_filename=<file>    The path to directory to write the output.
"""

import logging
logging.basicConfig(level=logging.INFO, filename='.log-parse-trees')
import os
import six
import sys
sys.path.insert(0, os.path.abspath('..'))
import utils

from nltk.parse.stanford import StanfordParser
from tqdm import tqdm


def main():
    """Main function of script."""
    args = utils.read_arguments(__doc__)

    # Read dataset. Each row of x_matrix is a sentence.
    x_matrix, y_vector = utils.pickle_from_file(args['input_filename'])

    # Get Stanford model
    parser = StanfordParser(
        model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',
        encoding='utf8')
    # Get parse trees.
    parsed_matrix = []
    for index, sentence in tqdm(enumerate(x_matrix), total=len(x_matrix)):
        try:
            parsed_matrix.append(list(parser.raw_parse(six.text_type(
                sentence.decode('utf-8')))))
        except UnicodeDecodeError:
            logging.warning('Skip sentence {} for unicode error'.format(index))
            y_vector.pop(index)

    # Save output
    utils.pickle_to_file((parsed_matrix, y_vector), args['output_filename'])


if __name__ == '__main__':
    main()
