"""Tests for the ArgBiLSTM model"""

import numpy
import unittest

from models import arg_bilstm

class ArgBiLSTMTest(unittest.TestCase):
    """Tests for the ArgBiLSTM model"""
    MODEL = arg_bilstm.ArgBiLSTM

    def setUp(self):
        self.batch_size = 10
        self.embedding_size = 5
        vocab_size = 12
        embeddings = numpy.random.rand(vocab_size, self.embedding_size)
        mappings = {
            'tokens': {'hello': 1, 'world': 2, 'it': 3, 'is': 5, 'nice': 4,
                       'a': 6, 'day': 7},
            'labels': {'0': 0, '1': 1, '2': 2},
            'casing': {
                'PADDING': 0, 'allLower': 4, 'allUpper': 5, 'contains_digit': 7,
                'initialUpper': 6, 'mainly_numeric': 3, 'numeric': 2,
                'other': 1
            }
        }
        data = {'name' : {
            'trainMatrix': [{
                    'labels': [0, 0],
                    'raw_tokens': ['hello', 'world'],
                    'tokens': [1, 2],
                    'casing': [5, 4]
                },{
                    'labels': [0, 0, 1, 1, 1],
                    'raw_tokens': ['it', 'is', 'a', 'nice', 'day'],
                    'tokens': [3, 5, 6, 4, 7],
                    'casing': [4, 4, 4, 4, 4]
                },{
                    'labels': [0, 0],
                    'raw_tokens': ['hello', 'world'],
                    'tokens': [1, 2],
                    'casing': [5, 4]
                }
            ],
            'devMatrix': [{
                    'labels': [1, 1, 0, 0],
                    'raw_tokens': ['nice', 'day', 'it', 'is'],
                    'tokens': [4, 7, 3, 5],
                    'casing': [4, 4, 4, 4]
                }],
            'testMatrix': [{
                    'labels': [1, 1, 0, 0],
                    'raw_tokens': ['nice', 'day', 'it', 'is'],
                    'tokens': [4, 7, 3, 5],
                    'casing': [4, 4, 4, 4]
                }]
        }}
        datasets = {'name': {
            'columns': {},  # Not necessary for test
            'commentSymbol': None,  # Not necessary for test
            'dirpath': '',  # Not necessary for test
            'evaluate': True,
            'label': 'labels'
        }}
        classifier_params = {
            'charEmbeddingsSize': None,
            'charEmbeddings': None
        }

        self.model = self.MODEL(classifier_params)
        self.model.setMappings(mappings, embeddings)
        self.model.setDataset(datasets, data)

    def test_build(self):
        """Test the model can be successfully built"""
        self.model.buildModel()

    def test_fit(self):
        """Test the model can be fitted"""
        self.model.fit(epochs=2)


class AttArgBiLSTMTest(ArgBiLSTMTest):
    MODEL = arg_bilstm.AttArgBiLSTM


if __name__ == '__main__':
    unittest.main()