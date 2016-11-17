import copy
import mock
import unittest

import evaluation

class ClassifierMock(object):
    def __init__(self):
        self.params = {'1': 0, '2': 0}
        self.parameters_used = []

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def fit_transform(self, dummy1, dummy2):
        print 'Fitting!', self.params
        self.parameters_used.append(copy.copy(self.params))

    def predict(self, _):
        return []


class TestGrid(unittest.TestCase):

    @mock.patch('evaluation.log_parameters')
    def test_grid(self, log_mock):
        parameters = {'1': [1, 2, 3], '2':[2, 3, 4]}
        classifier = ClassifierMock()
        evaluation.run_grid_search([], [], [], [], classifier, parameters)

        combinations = 9
        print classifier.parameters_used
        self.assertEqual(combinations, len(classifier.parameters_used))
        expected_combinations = [
            [{'1': 1}, {'2': 2}], [{'1': 1}, {'2': 3}], [{'1': 1}, {'2': 4}],
            [{'1': 2}, {'2': 2}], [{'1': 2}, {'2': 3}], [{'1': 2}, {'2': 4}],
            [{'1': 3}, {'2': 2}], [{'1': 3}, {'2': 3}], [{'1': 3}, {'2': 4}],
        ]
        for combination in expected_combinations:
            self.assertIn(combination, classifier.parameters_used)

if __name__ == '__main__':
    unittest.main()
