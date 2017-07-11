"""Auxiliary functions."""

import pickle
import docopt
import os
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


def safe_mkdir(directory_name):
    """Creates a directory only if it doesn't exists"""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)