"""Auxiliary functions."""

try:
    import cPickle as pickle
except ImportError:
    import pickle
import docopt
import re


def pickle_to_file(object_, filename):
    """Abstraction to pickle object with the same protocol always."""
    file_ = open(filename, 'wb')
    pickle.dump(object_, file_, pickle.HIGHEST_PROTOCOL)
    file_.close()


def pickle_from_file(filename):
    """Abstraction to read pickle file with the same protocol always."""
    with open(filename, 'rb') as file_:
        return pickle.load(file_)


def read_arguments(doc):
    """Reads the arguments values from stdin."""
    raw_arguments = docopt.docopt(doc)
    arguments = {re.sub(r'[-,<,>,]', '', key): value
                 for key, value in raw_arguments.items()}
    return arguments
