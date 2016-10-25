"""Functions to evaluates a classifier"""

import logging
logging.basicConfig(level=logging.INFO)
import numpy

from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm import tqdm


def evaluate(x_matrix, y_vector, classifier, folds=5):
    scores = cross_val_score(classifier, x_matrix, y_vector,
                             scoring='f1_macro', cv=folds)
    logging.info('f1_macro with Tfidf: %0.2f (+/- %0.2f)' % (
        scores.mean(), scores.std() * 2))


def deep_evaluate(x_matrix, y_vector, classifier, folds=5):
    """Prints the classification report for folds tratified split of x_matrix.
    """
    skf = StratifiedKFold(n_splits=folds)
    logging.info('Deep evaluating classifier {}'.format(classifier))
    x_matrix = numpy.array(x_matrix).squeeze()  # Removes extra dimensions
    y_vector = numpy.array(y_vector)
    for train_index, test_index in tqdm(skf.split(x_matrix, y_vector),
            total=folds):
        logging.info('Iteration:')
        x_train, x_test = x_matrix[train_index], x_matrix[test_index]
        y_train, y_test = y_vector[train_index], y_vector[test_index]
        classifier.fit_transform(x_train, y_train)
        logging.info('Performance in test dataset')
        predictions = classifier.predict(x_test)
        log_report(predictions, y_test)
        logging.info('Performance in train dataset')
        predictions = classifier.predict(x_train)
        log_report(predictions, y_train)


def log_report(predictions, y_test):
    """Logs the classification_report and confusion matrix."""
    target_names = ['Claim', 'MajorClaim', 'None', 'Premise']
    logging.info('Classification report')
    logging.info(classification_report(y_test, predictions, digits=3,
                                       target_names=target_names))
    logging.info('Confusion matrix')
    for row in confusion_matrix(y_test, predictions):
        logging.info('\t'.join([str(count) for count in row]))


def evaluate_grid_search(x_matrix, y_vector, classifier, parameters):
    x_train, x_test, y_train, y_test = train_test_split(
        x_matrix, y_vector, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(classifier, parameters, n_jobs=-1,
                               scoring='f1_macro', verbose=1)
    grid_search.fit(x_train, y_train)
    logging.info('---- New grid search ----')
    logging.info('Grid search best score {}'.format(grid_search.best_score_))
    logging.info('Grid search best params:')
    for param_name in sorted(parameters.keys()):
        logging.info('%s: %r' % (param_name,
                                 grid_search.best_params_[param_name]))
    predictions = grid_search.predict(x_test)
    log_report(predictions, y_test)
