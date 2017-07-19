"""Script to preprocess the Argumentative Essays dataset.

The output is a pickled tuple.
If --raw_text is present, the first element is a 2D numeric matrix, where
instances are separated by document and by sentence. If --raw_text is not
present, the first element of the tuple is a list of AnnotatedDocuments.
The second is a 2D matrix
with the labels for each sentence.

Usage:
    process_arg_documents.py --input_dirpath=<dirpath> --output_filename=<filename> [--raw_text] [--limit=<N>]

Options:
    --input_dirpath=<dirpath>        The path to directory to read files.
    --output_filename=<filename>     The path to directory to store files.
    --limit=<N>                  The number of files to read. -1 for all. [default: -1]
    --raw_text                       Save only sentences without process.
"""

import logging
logging.basicConfig(level=logging.INFO)
import nltk
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

    def label_start_position(self, start, end):
        """Returns the position of a label between start and end, or -1."""
        start_index = -1
        labels_indexes = sorted(self.raw_labels.keys())
        for index in labels_indexes:
            if index >= start and index < end:
                start_index = index
                break
            if index > end:
                break
        return start_index

    def _get_label(self, sentence, current_start_index):
        """Returns the label for sentence"""
        # Sentence has a label
        start_index = self.label_start_position(
            current_start_index, current_start_index + len(sentence))
        if start_index < 0:
            return 'None'
        component_start = start_index - current_start_index
        label, component_text = self.raw_labels[start_index]
        if len(component_text) <= len(sentence[component_start:]):
            assert sentence.find(component_text) >= 0
        else:  # The component text extends to next sentence
            # Next label starts with the next sentence
            next_start = current_start_index + len(sentence) + 1
            # Add again the label for next sentence, and only
            # the relevant part of the component_text
            self.raw_labels[next_start] = (
                label, component_text[len(sentence) - component_start + 1:])
        return label

    def get_labeled_sentences(self):
        """Returns all instances and corresponding labels in two lists."""
        self._get_labels()
        essay_text = self.instance_input_file.read()
        paragraphs = essay_text.split('\n')
        current_start_index = len(paragraphs[0]) + 1  # New lines
        labels = []
        splited_paragraphs = []
        for paragraph in paragraphs[2:]:  # Skip essay title
            current_start_index += 1  # New line at the end of last paragraph
            if not len(paragraph):
                continue
            sentences = nltk.tokenize.sent_tokenize(paragraph)
            sentence_labels = []
            for sentence in sentences:
                sentence_labels.append(self._get_label(sentence,
                                                       current_start_index))
                assert essay_text[current_start_index:current_start_index+len(sentence)] == sentence
                current_start_index += len(sentence)
                if (current_start_index < len(essay_text) and
                        essay_text[current_start_index] == ' '):
                    current_start_index += 1
            assert len(sentences) == len(sentence_labels)
            labels.append(sentence_labels)
            splited_paragraphs.append(sentences)
        return splited_paragraphs, labels


class FeatureExtractor(object):
    """Converts a list of natural text sentences an labels to numeric matrix.
    """
    def __init__(self, ngrams_degree=NGRAM_DEGREE):
        self.vectorizer = DictVectorizer()
        self.ngrams = []
        self.ngrams_degree = ngrams_degree

    def get_matrix(self, sentences):
        """Extracts features from list of sentences."""
        for document in sentences:
            for paragraph in document:
                for sentence in paragraph:
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


def get_input_files(input_dirpath, pattern, limit=-1):
    """Returns the names of the files in input_dirpath that matches pattern."""
    all_files = os.listdir(input_dirpath)
    if limit < 0:
        limit = len(all_files)
    for filename in all_files[:limit]:
        if re.match(pattern, filename) and os.path.isfile(os.path.join(
                input_dirpath, filename)):
            yield os.path.join(input_dirpath, filename)


def main():
    """Main fuction of the script."""
    args = utils.read_arguments(__doc__)
    sentences = []
    labels = []
    for filename in get_input_files(args['input_dirpath'], r'.*txt',
                                    args['limit']):
        with LabeledSentencesExtractor(filename) as instance_extractor:
            labeled_senteces = instance_extractor.get_labeled_sentences()
            sentences.append(labeled_senteces[0])
            labels.append(labeled_senteces[1])

    if not args['raw_text']:
        # Process sentences as vectors
        feature_extractor = FeatureExtractor()
        x_train = feature_extractor.get_matrix(sentences)
        logging.info('Saving numeric matrix with shape {}'.format(
            x_train.shape))
    else:
        x_train = sentences
        logging.info('Saving raw text, {} documents'.format(len(sentences)))

    # Convert labels to numeric vector
    unique_labels = sorted(['Claim', 'MajorClaim', 'Premise', 'None'])
    counts = dict.fromkeys(unique_labels, 0)
    y_vector = []
    for document_labels in labels:
        for paragraph_labels in document_labels:
            for label_index, label in enumerate(paragraph_labels):
                paragraph_labels[label_index] = unique_labels.index(label)
                counts[label] += 1
    logging.info('Classes used (sorted) {}'.format(unique_labels))
    logging.info('\t Counts {}'.format(counts))

    utils.pickle_to_file((x_train, labels), args['output_filename'])



if __name__ == '__main__':
    main()
