"""Tests for the SelfAttArgBiLSTM model"""

import numpy
import unittest

from models.selfatt_arg_bilstm import SelfAttArgBiLSTM
from models.test_arg_bilstm import ArgBiLSTMTest

class SelfAttArgBiLSTMTest(ArgBiLSTMTest):
    MODEL = SelfAttArgBiLSTM


if __name__ == '__main__':
    unittest.main()
