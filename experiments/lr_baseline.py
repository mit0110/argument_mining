"""Baseline with support vector machine.

Usage:
    lr_baseline.py --input_filename=<filename> [--use_trees]

Options:
   --input_filename=<filename>      The path to directory to read the dataset.
   --use_trees                      The input is a pickled parse tree.
"""

import logging
logging.basicConfig(level=logging.INFO, filename='.log-lr')
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import evaluation
import process_pipeline
import utils

from sklearn.linear_model import LogisticRegression

def main():
    """Script main function"""
    args = utils.read_arguments(__doc__)

    # Read dataset
    x_matrix, y_vector = utils.pickle_from_file(args['input_filename'])

    if args['use_trees']:
        classifier = process_pipeline.get_basic_tree_pipeline(
            ('clf', LogisticRegression(C=1, n_jobs=-1)))
        parameters = process_pipeline.get_tree_parameter_grid()
        parameters['clf__C'] = (1, 0.5, 0.3, 0.1, 0.05)
    else:
        classifier = process_pipeline.get_basic_pipeline(
            ('clf', LogisticRegression(C=1, n_jobs=-1)))
        classifier.set_params(**{  # Optimized for LR
            'features__vect__max_features': 1000,
            'features__vect__ngram_range': (1, 1)
        })
        parameters = process_pipeline.get_basic_parameters()
        for parameter in parameters:
            parameter['clf__C'] = (1, 0.5, 0.3, 0.1, 0.05, 1e-2, 1e-3)

    evaluation.evaluate_grid_search(x_matrix, y_vector, classifier, parameters)


if __name__ == '__main__':
    main()
