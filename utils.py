"""Auxiliary functions."""

import cPickle
import docopt
import re


def pickle_to_file(object_, filename):
    """Abstraction to pickle object with the same protocol always."""
    file_ = open(filename, 'w')
    cPickle.dump(object_, file_, cPickle.HIGHEST_PROTOCOL)
    file_.close()


def pickle_from_file(filename):
    """Abstraction to read pickle file with the same protocol always."""
    with open(filename, 'r') as file_:
        return cPickle.load(file_)


def read_arguments(doc):
    """Reads the arguments values from stdin."""
    raw_arguments = docopt.docopt(doc)
    arguments = {re.sub(r'[-,<,>,]', '', key): value
                 for key, value in raw_arguments.iteritems()}
    return arguments
