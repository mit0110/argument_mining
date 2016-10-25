"""Script to preprocess the Argumentative Essays dataset.

The output is a numeric matrix stored in 4 files:
x_train.pickle The numeric matrix to use for training a classifier.
y_train.pickle The true labels of each element in x_train.
x_test.pickle The numeric matrix to use for training a classifier.
y_test.pickle The true labels of each element in x_train.

Usage:
    process_arg_essays.py --input_dirpath=<dirpath> --output_filename=<filename> [--raw_text]

Options:
   --input_dirpath=<dirpath>        The path to directory to read files.
   --output_filename=<filename>     The path to directory to store files.
   --raw_text                       Save only sentences without process.
"""

import logging
logging.basicConfig(level=logging.INFO)
import nltk
import numpy
import re
import os

import sys
sys.path.insert(0, os.path.abspath('..'))

import utils

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
        """Read labels from the annotation file.

        Each label is a map
            start_index -> (label type, component text).
        Saves the resulting labels in self.raw_labels
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

    def get_labeled_sentences(self):
        """Returns all instances and corresponding labels in two lists."""
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
        assert len(sentences) == len(labels) + 1
        return sentences[1:], labels


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


def main():
    """Main fuction of the script."""
    args = utils.read_arguments(__doc__)
    sentences = []
    labels = []
    for filename in get_input_files(args['input_dirpath'], r'.*txt'):
        with LabeledSentencesExtractor(filename) as instance_extractor:
            labeled_senteces = instance_extractor.get_labeled_sentences()
            sentences.extend(labeled_senteces[0])
            labels.extend(labeled_senteces[1])

    if not args['raw_text']:
        # Process sentences as vectors
        feature_extractor = FeatureExtractor()
        x_train = feature_extractor.get_matrix(sentences)
        logging.info('Saving numeric matrix with shape {}'.format(
            x_train.shape))
    else:
        x_train = sentences
        logging.info('Saving raw text, {} sentences'.format(len(sentences)))

    # Convert labels to numeric vector
    unique_labels = sorted(numpy.unique(labels).tolist())
    y_vector = [unique_labels.index(label) for label in labels]
    logging.info('Classes used (sorted) {}'.format(unique_labels))

    utils.pickle_to_file((x_train, y_vector), args['output_filename'])



if __name__ == '__main__':
    main()
