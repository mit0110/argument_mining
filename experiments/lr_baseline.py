"""Baseline with support vector machine.

Usage:
    lr_baseline.py --input_filename=<filename>

Options:
   --input_filename=<filename>      The path to directory to read the dataset.
"""

import logging
logging.basicConfig(level=logging.INFO, filename='.log-lr')
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import evaluation
import utils

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def main():
    """Script main function"""
    args = utils.read_arguments(__doc__)

    # Read dataset
    x_matrix, y_vector = utils.pickle_from_file(args['input_filename'])

    classifier = Pipeline([
        ('vect', CountVectorizer(
            ngram_range=(1, 3), max_features=10**4)),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(C=0.01, n_jobs=-1)),
    ])
    evaluation.evaluate(x_matrix, y_vector, classifier)

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_features': [10**3, 10**4, 10**5],
        'tfidf__use_idf': (True, False),
        'clf__C': (1, 0.5, 0.3, 0.1, 0.05, 1e-2, 1e-3),
    }
    grid_search = GridSearchCV(
        classifier, parameters, n_jobs=-1, scoring='f1_macro')
    evaluation.evaluate_grid_search(x_matrix, y_vector, grid_search, parameters)


if __name__ == '__main__':
    main()