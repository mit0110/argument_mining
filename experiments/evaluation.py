"""Functions to evaluates a classifier"""

import copy
import json
import logging
logging.basicConfig(level=logging.INFO)
import numpy
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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
        logging.info('Performance in train dataset')
        predictions = classifier.predict(x_train)
        log_report(predictions, y_train)
        scores['train'].append(precision_recall_fscore_support(
            predictions, y_train, average='macro')[:3])
        logging.info('Performance in test dataset')
        predictions = classifier.predict(x_test)
        log_report(predictions, y_test)
        scores['test'].append(precision_recall_fscore_support(
            predictions, y_test, average='macro')[:3])

    logging.info('Train average scores: {}'.format(
        numpy.array(scores['train']).mean(axis=0)))
    logging.info('Train scores std: {}'.format(
        numpy.array(scores['train']).std(axis=0)))

    logging.info('Test average scores: {}'.format(
        numpy.array(scores['test']).mean(axis=0)))
    logging.info('Test scores std: {}'.format(
        numpy.array(scores['test']).std(axis=0)))


def log_report(predictions, y_test):
    """Logs the classification_report and confusion matrix."""
    target_names = ['Claim', 'MajorClaim', 'None', 'Premise']
    logging.info('\nClassification report')
    logging.info(classification_report(y_test, predictions, digits=3,
                                       target_names=target_names))
    logging.info('Macro metrics: {}'.format(precision_recall_fscore_support(
            predictions, y_test, average='macro')))
    logging.info('\nConfusion matrix')
    for row in confusion_matrix(y_test, predictions):
        logging.info('\t'.join([str(count) for count in row]))


def evaluate_grid_search(x_matrix, y_vector, classifier, parameters,
                         log_file=None):
    grid = GridSearchCustom(
        complex_params=['features__flat__extractors__transformer_list'],
        process_y_function=process_pipeline.transform_y)
    score, parameters = grid.search(x_matrix, y_vector, classifier, parameters)
    grid.log_parameters(parameters, score)
    logging.info('Best score {}'.format(score))
    logging.info('Best parameters: {}'.format(grid.log[-1][0]))
    if log_file:
        grid.save_log(log_file)
    grid.retrain()


class GridSearchCustom(object):
    """Custom exahustive grid search class."""
    def __init__(self, complex_params=[], process_y_function=None):
        self.iterations = 0
        self.total_iterations = 0
        self.complex_params = complex_params
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.classifier = None
        self.log = []
        if not process_y_function:
            self.process_y_function = lambda x: x
        else:
            self.process_y_function = process_y_function
        self.recursion_level = 0
        self.best_parameters = {}

    def search(self, x_matrix, y_vector, classifier, parameters):
        """Performs exahustive search over parameters."""
        numpy.set_printoptions(suppress=True)
        self.total_iterations = 1
        for values in parameters.values():
            self.total_iterations = self.total_iterations * len(values)
        print 'Recursive class: {}'.format(self.total_iterations)
        logging.info('Recursive class: {}'.format(self.total_iterations))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x_matrix, y_vector, test_size=0.33)
        self.y_train = self.process_y_function(self.y_train)
        self.y_test = self.process_y_function(self.y_test)

        self.classifier = classifier
        score, best_parameters = self._run_grid_search(parameters)
        self.best_parameters = best_parameters
        return score, best_parameters

    def _run_grid_search(self, parameters, used_parameters=None, max_score=0):
        self.recursion_level += 1
        if not used_parameters:
            used_parameters = {}
        # Base case
        if parameters == {}:
            self.iterations += 1
            print 'iteration {}/{}'.format(self.iterations,
                                           self.total_iterations)
            score = self.get_f1_score()
            self.log_parameters(used_parameters, score)
            self.recursion_level -= 1
            if max_score < score:
                return score, copy.copy(used_parameters)
            return None

        best_parameters = {}

        parameter, values = parameters.popitem()
        for value in values:
            self.classifier.set_params(**{parameter: value})
            used_parameters[parameter] = value
            result = self._run_grid_search(
                copy.copy(parameters), used_parameters, max_score)
            if result:
                max_score, best_parameters = result
            # import ipdb; ipdb.set_trace()
        self.recursion_level -= 1
        if best_parameters != {}:
            return max_score, best_parameters
        return None

    def get_f1_score(self):
        self.classifier.fit(self.x_train, self.y_train)
        predictions = self.classifier.predict(self.x_test)
        return f1_score(self.y_test, predictions, average='macro')

    def log_parameters(self, parameters, score):
        used_parameters = {}
        for parameter, values in parameters.iteritems():
            if parameter in self.complex_params:
                used_parameters[parameter] = [feature[0] for feature in values]
            else:
                used_parameters[parameter] = values
        self.log.append((used_parameters, score))

    def save_log(self, output_file):
        logging.info('Saving log file')
        with open(output_file, 'w') as output_file:
            json.dump(self.log, output_file)

    def retrain(self):
        """Retrains the classifier with the best parameters"""
        for parameter, values in self.best_parameters.iteritems():
            self.classifier.set_params(**{parameter: values})
        self.classifier.fit(self.x_train, self.y_train)
        logging.info('Training Performance')
        predictions = self.classifier.predict(self.x_train)
        log_report(predictions, self.y_train)
        logging.info('Testing Performance')
        predictions = self.classifier.predict(self.x_test)
        log_report(predictions, self.y_test)
