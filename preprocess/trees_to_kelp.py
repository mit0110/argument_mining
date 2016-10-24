"""Script to transform nltk parse trees to kelp format.

Usage:
    trees_to_kelp.py --input_filename=<filename> --output_filename=<filename>

Options:
   --input_filename=<filename>  The path to directory to read the dataset.
   --output_filename=<filename> The path to directory to store the result.

"""

import logging
logging.basicConfig(level=logging.INFO, filename='.log-trees2kelp')
import re
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import utils


def tree_to_line(tree):
    """Returns the formated representation of the tree."""
    stree = tree.pformat()
    return re.sub( '\s+', ' ', stree).strip()


def instance_line(tree, label):
    """Returns the formated representation of the instance."""
    return '{} |BT:tree| {} |ET|'.format(label, tree_to_line(tree))


def main():
    """Main function of script."""
    args = utils.read_arguments(__doc__)

    # Read dataset
    x_matrix, y_vector = utils.pickle_from_file(args['input_filename'])

    with open(args['output_filename'], 'w') as output_file:
        for tree, label in zip(x_matrix, y_vector):
            output_file.write(instance_line(tree[0], label) + '\n')


if __name__ == '__main__':
    main()
