"""Script to preprocess the Argumentative Essays dataset.

The output is a numeric matrix stored in 4 files:
x_train.pickle The numeric matrix to use for training a classifier.
y_train.pickle The true labels of each element in x_train.
x_test.pickle The numeric matrix to use for training a classifier.
y_test.pickle The true labels of each element in x_train.

Usage:
    process_arg_essays.py --input_dirpath=<dirpath>

Options:
   --input_dirpath=<dirpath>    The path to directory where to store files.
"""

import docopt
import re
import os


class InstanceExtractor(object):
    """Retrieves an intance from input_file."""
    def __init__(self, input_filename):
        self.input_filename = input_filename

    def __enter__(self):
        self.input_file = open(self.input_filename, 'r')

    def __exit__(self, exc_type, exc_value, traceback):
        self.input_file.close()

    def get_instance(self):
        """Returns next instance"""
        yield 'lala'


class DatasetHandler(object):
    """Abstraction to read and write datasets as numeric matrixes to files."""
    def __init__(self, dirpath, **kwargs):
        self.dirpath = dirpath
        if 'split_sizes' in kwargs:
            self.split_sizes = kwargs['split_sizes']
        else:
            self.split_sizes = [0.8, 0.2]

    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def x_test(self):
        return self._x_test

    @property
    def y_test(self):
        return self._y_test

    def save():
        pass

    def read():
        pass

    def build_from_matrix(matrix, labels):
        """Constructs a dataset from matrix and labels object."""
        pass



def get_input_files(input_dirpath, pattern):
    """Returns the names of the files in input_dirpath that matches pattern."""
    all_files = os.listdir(input_dirpath)
    result = []
    for filename in all_files:
        if re.match(pattern, filename) and os.path.isfile(os.path.join(
            input_dirpath, filename)):
            yield os.path.join(input_dirpath, filename)


def read_arguments():
    """Reads the arguments values from stdin."""
    raw_arguments = docopt.docopt(__doc__)
    arguments = {re.sub(r'[-,<,>,]', '', key): value
                 for key, value in raw_arguments.iteritems()}
    return arguments


def main():
    """Main fuction of the script."""
    args = read_arguments()

    for filename in get_input_files(args.input_dirpath, r'.*txt'):
        with InstanceExtractor(filename) as instance_extractor:
            for instance, label in instance_extractor:
                # Save instance and label
                break



if __name__ == '__main__':
    main()
