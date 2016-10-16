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
import nltk
import re
import os

from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer


NGRAM_DEGREE = 3


class LabeledSentencesExtractor(object):
    """Retrieves an intance from input_file."""
    def __init__(self, input_filename):
        self.input_filename = input_filename
        self.label_input_file = None
        self.instance_input_file = None
        self.raw_labels = {}

    def __enter__(self):
        self.label_input_file = open(
            self.input_filename.replace('txt', 'ann'), 'r')
        self.instance_input_file = open(self.input_filename, 'r')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.label_input_file.close()
        self.instance_input_file.close()

    def _get_labels(self):
        """Read labels from file.

        Each label is a map
            start_index -> (label type, component text).
        """
        self.raw_labels = {}
        for line in self.label_input_file.readlines():
            if line == '' or not line.startswith('T'):
                continue
            label_info, text = line.split('\t')[1:3]  # Remove the first element
            label, start_index, _ = label_info.split()
            self.raw_labels[int(start_index)] = (label, text.strip())

    def _get_label(self, sentence, labels_indexes, current_start_index):
        """Returns the label for sentence"""
        # Sentence has a label
        if not (labels_indexes[0] >= current_start_index and
                labels_indexes[0] < current_start_index + len(sentence)):
            return 'None'
        start_index = labels_indexes.pop(0)
        component_start = start_index - current_start_index
        label, component_text = self.raw_labels[start_index]
        if len(component_text) <= len(sentence[:component_start]):
            assert sentence.find(component_text) > 0
        else:  # The component text extends to next sentence
            # Next label starts with the next sentence
            next_start = current_start_index + len(sentence) + 1
            labels_indexes.insert(0, next_start)
            # Add again the label for next sentence, and only
            # the relevant part of the component_text
            self.raw_labels[next_start] = (label, component_text[
                :len(sentence) - component_start])
        return label

    def get_instance(self):
        """Returns next instance"""
        self._get_labels()
        essay_text = self.instance_input_file.read()
        sentences = nltk.tokenize.sent_tokenize(essay_text)
        current_start_index = len(sentences[0]) + 2  # Two new lines
        labels_indexes = sorted(self.raw_labels.keys())
        labels = []
        for sentence in sentences[1:]:  # Skip essay title
            labels.append(self._get_label(sentence, labels_indexes,
                                          current_start_index))
            current_start_index += len(sentence) + 1  # New line at the end.
        return sentences, labels


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


class FeatureExtractor(object):
    """Converts a list of natural text sentences an labels to numeric matrix.
    """
    def __init__(self, ngrams_degree=NGRAM_DEGREE):
        self.vectorizer = DictVectorizer()
        self.ngrams = []
        self.ngrams_degree = ngrams_degree

    def get_matrix(self, sentences):
        """Extracts features from list of sentences."""
        for sentence in sentences:
            words = nltk.tokenize.word_tokenize(sentence)
            self.ngrams.append(self.count_ngrams(words))
        matrix = self.vectorizer.fit_transform(self.ngrams)
        return matrix

    def count_ngrams(self, word_list):
        """Extract all counts n-grams where n <= max_degree."""
        sentence_ngrams = defaultdict(lambda: 0)
        for degree in range(1, self.ngrams_degree + 1):
            for ngram in zip(*[word_list[i:] for i in range(degree)]):
                if isinstance(ngram, tuple):
                    ngram = ' '.join(ngram)
                sentence_ngrams[ngram] += 1
        return dict(sentence_ngrams)


def get_input_files(input_dirpath, pattern):
    """Returns the names of the files in input_dirpath that matches pattern."""
    all_files = os.listdir(input_dirpath)
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

    for filename in get_input_files(args['input_dirpath'], r'.*txt'):
        with LabeledSentencesExtractor(filename) as instance_extractor:
            sentences, labels = instance_extractor.get_labeled_sentences()

    # Process sentences as vectors
    feature_extractor = FeatureExtractor()
    x_matrix = feature_extractor.get_matrix(sentences)
    # Vectors



if __name__ == '__main__':
    main()
