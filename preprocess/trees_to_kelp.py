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

from nltk.corpus.reader.wordnet import POS_LIST
from nltk.stem import WordNetLemmatizer


def tree_to_line(tree, stemmer):
    """Returns the formated representation of the tree."""
    # Steam all words
    # Get tree representation
    stree = tree.pformat()
    stree = re.sub( '\s+', ' ', stree).strip()

    for word, pos in tree.pos():
        if pos[0].lower() in POS_LIST:
            token = stemmer.lemmatize(word, pos=pos[0].lower())
        else:
            token = stemmer.lemmatize(word)
        stree = stree.replace(word, token)

    return stree


def instance_line(tree, label, stemmer):
    """Returns the formated representation of the instance."""
    return '{} |BT:tree| {} |ET|'.format(label, tree_to_line(tree, stemmer))


def main():
    """Main function of script."""
    args = utils.read_arguments(__doc__)

    # Read dataset
    x_matrix, y_vector = utils.pickle_from_file(args['input_filename'])
    stemmer = WordNetLemmatizer()

    with open(args['output_filename'], 'w') as output_file:
        for tree, label in zip(x_matrix, y_vector):
            output_file.write(instance_line(tree[0], label, stemmer) + '\n')


if __name__ == '__main__':
    main()
