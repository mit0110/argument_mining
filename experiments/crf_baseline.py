"""Train a CRF with conll formated dataset.

Usage:
    crf_baseline.py --input_filename=<filename> [--search_grid]

Options:
    --input_filename <filename>      The path to directory to read the dataset.
    --search_grid                    Make extensive classifier parameter search
"""
from __future__ import print_function

import logging
logging.basicConfig(level=logging.INFO, filename='logs/log-crf',
                    format='%(message)s')

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../preprocess/'))

import itertools
import evaluation
import conll_feature_extractor
import scipy
import sklearn_crfsuite
import utils

from sklearn.model_selection import train_test_split
from sklearn.grid_search import RandomizedSearchCV

from sklearn import metrics
from sklearn_crfsuite import metrics as suite_metrics


def main():
    """Main function of script"""
    args = utils.read_arguments(__doc__)
    documents = utils.pickle_from_file(args['input_filename'])

    transformer = conll_feature_extractor.ConllFeatureExtractor(
        use_structural=True, use_syntactic=True, # use_lexical=True
    )
    # Extract instances and labels. Each instance is a sentence, represented as
    # a list of feature dictionaries for each work. Labels are represented as
    # a list of word labels.
    instances = transformer.get_feature_dict(documents)
    labels = conll_feature_extractor.get_labels_from_documents(documents)

    x_train, x_test, y_train, y_test = train_test_split(instances, labels,
                                                        test_size=0.33)

    classifier = sklearn_crfsuite.CRF(
        algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100,
        all_possible_transitions=True
    )

    if not args['search_grid']:
        classifier.fit(x_train, y_train)
        predictions = list(itertools.chain(*classifier.predict(x_test)))

        evaluation.log_report(predictions, list(itertools.chain(*y_test)))
    else:
        # label_names = list(classifier.classes_)
        # label_names.remove('O')
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }
        f1_scorer = metrics.make_scorer(suite_metrics.flat_f1_score,
                                        average='weighted')#, labels=label_names)
        # search
        rs = RandomizedSearchCV(
            classifier, params_space, cv=3, verbose=1, n_jobs=-1, n_iter=50,
            scoring=f1_scorer)
        rs.fit(x_train, y_train)
        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        classifier = rs.best_estimator_
        predictions = list(itertools.chain(*classifier.predict(x_test)))
        evaluation.log_report(predictions, list(itertools.chain(*y_test)))


if __name__ == '__main__':
    main()
