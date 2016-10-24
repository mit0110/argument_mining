"""Baseline with support vector machine.

Usage:
    svm_baseline.py --input_filename=<filename> [--use_trees] [--search_grid]

Options:
   --input_filename=<filename>      The path to directory to read the dataset.
   --use_trees                      The input is a pickled parse tree.
   --search_grid                    Make extensive parameter search
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

    if args['use_trees']:
        classifier = process_pipeline.get_basic_tree_pipeline(
            ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                                  n_iter=5, random_state=42)))
        parameters = process_pipeline.get_tree_parameter_grid()
        features = process_pipeline.get_tree_steps()
        classifier.set_params(**{  # Optimized for SVM
            'features__ngrams__word_counter__max_features': 1000,
            'features__ngrams__word_counter__ngram_range': (1, 1),
            'features__transformer_list': [
                ('ngrams', features['ngrams']),
                ('pos_tags', features['pos_tags'])
            ],
        })
        
        # Grid parameters
        parameters['clf__alpha'] = (0.3, 0.1, 0.05, 1e-2, 1e-3)
    else:
        classifier = process_pipeline.get_basic_pipeline(
            ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-2,
                                  n_iter=5, random_state=42))
        )

        parameters = process_pipeline.get_basic_parameters()
        for parameter in parameters:
            parameter['clf__alpha'] = (0.3, 0.1, 0.05, 1e-2, 1e-3)


    if args['search_grid']:
        evaluation.evaluate_grid_search(x_matrix, y_vector,
                                        classifier, parameters)
    else:
        evaluation.deep_evaluate(x_matrix, y_vector, classifier)


if __name__ == '__main__':
    main()
