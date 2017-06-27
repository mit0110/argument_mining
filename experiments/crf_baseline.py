"""Train a CRF with conll formated dataset.

Usage:
    crf_baseline.py --input_filename=<filename>

Options:
   --input_filename <filename>      The path to directory to read the dataset.
"""

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
import sklearn_crfsuite
import utils

from sklearn.model_selection import train_test_split


def main():
    """Main function of script"""
    args = utils.read_arguments(__doc__)
    documents = utils.pickle_from_file(args['input_filename'])

    transformer = conll_feature_extractor.ConllFeatureExtractor()
    instances = transformer.get_feature_dict(documents)
    labels = conll_feature_extractor.get_labels_from_documents(documents)

    x_train, x_test, y_train, y_test = train_test_split(instances, labels,
                                                        test_size=0.33)

    classifier = sklearn_crfsuite.CRF(
        algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100,
        all_possible_transitions=True
    )

    classifier.fit(x_train, y_train)
    predictions = list(itertools.chain(*classifier.predict(x_test)))

    evaluation.log_report(predictions, list(itertools.chain(*y_test)))


if __name__ == '__main__':
    main()
