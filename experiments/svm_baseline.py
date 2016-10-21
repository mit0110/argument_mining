"""Baseline with support vector machine.

Usage:
    svm_baseline.py --input_filename=<filename>

Options:
   --input_filename=<filename>      The path to directory to read the dataset.
"""

import logging
logging.basicConfig(level=logging.INFO, filename='.log-svm')
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import evaluation
import process_pipeline
import utils

from sklearn.linear_model import SGDClassifier


def main():
    """Script main function"""
    args = utils.read_arguments(__doc__)

    # Read dataset
    x_matrix, y_vector = utils.pickle_from_file(args['input_filename'])

    classifier = process_pipeline.get_basic_pipeline(
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-2,
                              n_iter=5, random_state=42))
    )
    # evaluation.evaluate(x_matrix, y_vector, classifier)

    parameters = process_pipeline.get_basic_parameters()
    for parameter in parameters:
        parameter['clf__alpha'] = (0.3, 0.1, 0.05, 1e-2, 1e-3)
    evaluation.evaluate_grid_search(x_matrix, y_vector, classifier, parameters)

if __name__ == '__main__':
    main()
