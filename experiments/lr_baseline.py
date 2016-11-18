"""Baseline with support vector machine.

Usage:
    lr_baseline.py --input_filename=<filename> [--search_grid]

Options:
   --input_filename=<filename>      The path to directory to read the dataset.
   --search_grid                    Make extensive parameter search
"""

import logging
logging.basicConfig(level=logging.INFO, filename='logs/log-lr')
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import evaluation
import process_pipeline
import utils

from sklearn.linear_model import LogisticRegression


def get_optimized_params():
    features = process_pipeline.get_flat_features()
    return {  # Optimized for LR
        'features__flat__extractors__ngrams__word_counter__max_features': 10000,
        'features__flat__extractors__ngrams__word_counter__ngram_range': (1, 1),
        'features__flat__extractors__ngrams__tfidf__use_idf': False,
        'clf__C': 1,
        'features__flat__extractors__transformer_list': [
            ('ngrams', features['ngrams']),
            ('pos_tags', features['pos_tags']),
        ],
    }


def main():
    """Script main function"""
    args = utils.read_arguments(__doc__)

    # Read dataset
    x_matrix, y_vector = utils.pickle_from_file(args['input_filename'])

    classifier = process_pipeline.get_basic_tree_pipeline(
        ('clf', LogisticRegression(C=0.1, n_jobs=-1)))
    classifier.set_params(**get_optimized_params())

    if args['search_grid']:
        parameters = process_pipeline.get_tree_parameter_grid()
        parameters['clf__C'] = (1, 0.3, 0.1, 0.05, 0.01)
        evaluation.evaluate_grid_search(x_matrix, y_vector, classifier,
                                        parameters, log_file='logs/log-grid-lr')
    else:
        evaluation.deep_evaluate(x_matrix, y_vector, classifier)


if __name__ == '__main__':
    main()
