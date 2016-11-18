import copy
import mock
import unittest

import evaluation

class ClassifierMock(object):
    def __init__(self):
        self.params = {}
        self.parameters_used = []

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def fit(self, dummy1, dummy2):
        self.parameters_used.append(copy.copy(self.params))

    def predict(self, _):
        return sorted(self.params.values())


class TestGrid(unittest.TestCase):

    @mock.patch('evaluation.GridSearchCustom.log_parameters')
    def test_grid_2params(self, log_mock):
        parameters = {'1': [1, 2, 3], '2':[2, 3, 4]}
        classifier = ClassifierMock()
        grid = evaluation.GridSearchCustom()
        grid.y_test = [1, 2]
        grid.classifier = classifier
        grid._run_grid_search(parameters)

        combinations = 9
        self.assertEqual(combinations, len(classifier.parameters_used))
        expected_combinations = [
            {'1': 1, '2': 2}, {'1': 1, '2': 3}, {'1': 1, '2': 4},
            {'1': 2, '2': 2}, {'1': 2, '2': 3}, {'1': 2, '2': 4},
            {'1': 3, '2': 2}, {'1': 3, '2': 3}, {'1': 3, '2': 4},
        ]
        for combination in expected_combinations:
            self.assertIn(combination, classifier.parameters_used)

    @mock.patch('evaluation.GridSearchCustom.log_parameters')
    def test_grid_3params(self, log_mock):
        parameters = {'1': [1, 2], '2':[3, 4], '3': [5, 6]}
        classifier = ClassifierMock()
        grid = evaluation.GridSearchCustom()
        grid.y_test = [1, 3, 5]
        grid.classifier = classifier
        grid._run_grid_search(parameters)

        combinations = 8
        self.assertEqual(combinations, len(classifier.parameters_used))
        expected_combinations = [
            {'1': 1, '2': 3, '3': 5}, {'1': 1, '2': 4, '3': 6},
            {'1': 2, '2': 3, '3': 5}, {'1': 2, '2': 4, '3': 6},
            {'1': 2, '2': 4, '3': 5}, {'1': 2, '2': 3, '3': 6},
            {'1': 1, '2': 4, '3': 5}, {'1': 1, '2': 3, '3': 6},
        ]
        for combination in expected_combinations:
            self.assertIn(combination, classifier.parameters_used)

    @mock.patch('evaluation.GridSearchCustom.log_parameters')
    def test_grid_1param(self, log_mock):
        parameters = {'1': [1, 2]}
        classifier = ClassifierMock()
        grid = evaluation.GridSearchCustom()
        grid.y_test = [2]
        grid.classifier = classifier
        score, best_parameters = grid._run_grid_search(parameters)

        combinations = 2
        self.assertEqual(combinations, len(classifier.parameters_used))
        expected_combinations = [{'1': 1}, {'1': 2}]
        for combination in expected_combinations:
            self.assertIn(combination, classifier.parameters_used)
        self.assertEqual({'1': 2}, best_parameters)

    @mock.patch('evaluation.GridSearchCustom.log_parameters')
    def test_grid_best_params(self, log_mock):
        parameters = {'1': [1, 2, 3], '2':[4, 5]}
        classifier = ClassifierMock()
        grid = evaluation.GridSearchCustom()
        grid.y_test = [1, 4]
        grid.classifier = classifier
        score, best_parameters = grid._run_grid_search(parameters)

        # Perfect fit for params {'1': 1, '2': 3, '3': 5}
        self.assertEqual(1.0, score)

        self.assertEqual({'1': 1, '2': 4}, best_parameters)


    @mock.patch('evaluation.GridSearchCustom.log_parameters')
    def test_grid_best_3params(self, log_mock):
        parameters = {'1': [1, 2, 3], '2':[4, 5], '3': [5, 6]}
        classifier = ClassifierMock()
        grid = evaluation.GridSearchCustom()
        grid.y_test = [2, 4, 6]
        grid.classifier = classifier
        score, best_parameters = grid._run_grid_search(parameters)

        # Perfect fit for params {'1': 1, '2': 3, '3': 5}
        self.assertEqual(1.0, score)

        self.assertEqual({'1': 2, '2': 4, '3': 6}, best_parameters)


if __name__ == '__main__':
    unittest.main()
