"""Functions to evaluates a classifier"""

import copy
import logging
logging.basicConfig(level=logging.INFO)
import numpy

import process_pipeline

from sklearn.cross_validation import cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

from tqdm import tqdm


def deep_evaluate(x_matrix, y_vector, classifier, folds=5):
    """Prints the classification report for folds tratified split of x_matrix.
    """
    logging.info('Deep evaluating classifier {}'.format(classifier))
    x_matrix = numpy.array(x_matrix).squeeze()  # Removes extra dimensions
    y_vector = numpy.array(y_vector)
    scores = {'train': [], 'test': []}
    for iteration_index in tqdm(range(folds)):
        logging.info('Iteration: {}'.format(iteration_index))
        x_train, x_test, y_train, y_test = train_test_split(
            x_matrix, y_vector, test_size=0.33)
        y_train = process_pipeline.transform_y(y_train)
        y_test = process_pipeline.transform_y(y_test)
        classifier.fit_transform(x_train, y_train)
        logging.info('Performance in test dataset')
        predictions = classifier.predict(x_test)
        log_report(predictions, y_test)
        scores['test'].append(precision_recall_fscore_support(
            predictions, y_test, average='macro')[:3])
        logging.info('Performance in train dataset')
        predictions = classifier.predict(x_train)
        log_report(predictions, y_train)
        scores['train'].append(precision_recall_fscore_support(
            predictions, y_train, average='macro')[:3])

    logging.info('Train average scores: {}'.format(
        numpy.array(scores['train']).mean(axis=0)))
    logging.info('Test average scores: {}'.format(
        numpy.array(scores['test']).mean(axis=0)))


def log_report(predictions, y_test):
    """Logs the classification_report and confusion matrix."""
    target_names = ['Claim', 'MajorClaim', 'None', 'Premise']
    logging.info('\nClassification report')
    logging.info(classification_report(y_test, predictions, digits=3,
                                       target_names=target_names))
    logging.info('\nConfusion matrix')
    for row in confusion_matrix(y_test, predictions):
        logging.info('\t'.join([str(count) for count in row]))


def evaluate_grid_search(x_matrix, y_vector, classifier, parameters):
    x_train, x_test, y_train, y_test = train_test_split(
        x_matrix, y_vector, test_size=0.2, random_state=42)

    run_grid_search(x_train, x_test, y_train, y_test, classifier, parameters)

    # y_train = process_pipeline.transform_y(y_train)
    # y_test = process_pipeline.transform_y(y_test)
    # grid_search = GridSearchCV(classifier, parameters, n_jobs=-1,
    #                            scoring='f1_macro', verbose=1)
    # grid_search.fit(x_train, y_train)
    # logging.info('---- New grid search ----')
    # logging.info('Grid search best score {}'.format(grid_search.best_score_))
    # logging.info('Grid search best params:')
    # for param_name in sorted(parameters.keys()):
    #     logging.info('%s: %r' % (param_name,
    #                              grid_search.best_params_[param_name]))
    # predictions = grid_search.predict(x_test)
    # log_report(predictions, y_test)


def get_f1_score(x_train, x_test, y_train, y_test, classifier):
    y_train = process_pipeline.transform_y(y_train)
    y_test = process_pipeline.transform_y(y_test)
    classifier.fit_transform(x_train, y_train)
    predictions = classifier.predict(x_test)
    return f1_score(y_test, predictions)


def log_parameters(parameters):
    for parameter, values in parameters.iteritems():
        if not parameter == 'features__flat__extractors__transformer_list':
            print parameter, values
            logging.info('{}: {}'.format(parameter, values))
        else:
            print parameter, [feature[0] for feature in values]
            logging.info('{}: {}'.format(
                parameter, [feature[0] for feature in values]))


def run_grid_search(x_train, x_test, y_train, y_test, classifier, parameters,
                    used_parameters={}, max_score=0):
    if parameters == {}:
        logging.info('Fitting classifier')
        log_parameters(used_parameters)
        score = get_f1_score(x_train, x_test, y_train, y_test, classifier)
        score = 0
        logging.info('Score: {}'.format(score))
        return score, used_parameters

    best_parameters = {}

    combinations = 1
    for parameter, values in parameters.iteritems():
        combinations = combinations * len(values)
    print combinations

    while len(parameters) > 0:  # The last parameter has always being checked
        parameter, values = parameters.popitem()
        print 'Param', parameter, parameters
        for value in values:
            classifier.set_params(**{parameter: value})
            used_parameters[parameter] = value
            score, grid_parameters = run_grid_search(
                x_train, x_test, y_train, y_test, classifier, copy.copy(parameters),
                used_parameters, max_score)
            if max_score < score:
                max_score = score
                best_parameters =  grid_parameters

    return max_score, best_parameters
