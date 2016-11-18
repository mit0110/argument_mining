"""Train a CRF with conll formated dataset.

Usage:
    lr_baseline.py --input_filename=<filename> [--use_trees]

Options:
   --input_filename <filename>      The path to directory to read the dataset.
   --use_trees                      The input is a pickled parse tree.
"""

import logging
logging.basicConfig(level=logging.INFO, filename='logs/log-crf')

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import conll_feature_extractor
import utils

import sklearn_crfsuite


def main():
    """Main function of script"""
    args = utils.read_arguments(__doc__)
    documents, y_vector = utils.pickle_from_file(args['input_filename'])

    transformer = conll_feature_extractor.ConllFeatureExtractor()
    instances = transformer.get_feature_dict(documents)

    clf = sklearn_crfsuite.CRF(
        algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100,
        all_possible_transitions=True
    )

    clf.fit(instances, y_vector)


if __name__ == '__main__':
    main()
