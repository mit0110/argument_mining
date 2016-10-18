"""Functions to evaluates a classifier"""

import logging
logging.basicConfig(level=logging.INFO)

from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(x_matrix, y_vector, classifier):
    scores = cross_val_score(classifier, x_matrix, y_vector, cv=5)
    logging.info('Accuracy with Tfidf: %0.2f (+/- %0.2f)' % (
                 scores.mean(), scores.std() * 2))


def evaluate_grid_search(x_matrix, y_vector, grid_search, parameters):
    x_train, x_test, y_train, y_test = train_test_split(
        x_matrix, y_vector, test_size=0.2, random_state=42)

    grid_search.fit(x_train, y_train)
    logging.info('Grid search best score {}'.format(grid_search.best_score_))
    logging.info('Grid search best params:')
    for param_name in sorted(parameters.keys()):
        logging.info('%s: %r' % (param_name,
                                 grid_search.best_params_[param_name]))

    predictions = grid_search.predict(x_test)
    target_names = ['Claim', 'MajorClaim', 'None', 'Premise']
    logging.info('Classification report')
    logging.info(classification_report(y_test, predictions,
                                       target_names=target_names))
    logging.info('Confusion matrix')
    for row in confusion_matrix(y_test, predictions):
        logging.info('\t'.join([str(count) for count in row]))