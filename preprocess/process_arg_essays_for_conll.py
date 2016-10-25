"""Script to preprocess the Argumentative Essays dataset.

The output is a numeric matrix stored in 4 files:
x_train.pickle The numeric matrix to use for training a classifier.
y_train.pickle The true labels of each element in x_train.
x_test.pickle The numeric matrix to use for training a classifier.
y_test.pickle The true labels of each element in x_train.

Each instance is a sequence of words (document), and each label is list of
BIO tags. Each tag corresponds in order to the words in the document.
The tags also indicates the type of component: B-mc, B-c, B-p, O (none).

Usage:
    process_arg_essays_for_conll.py --input_dirpath=<dirpath> --output_file=<file> [--limit=<N>]

Options:
   --input_dirpath=<dirpath>    The path to directory to read files.
   --output_file=<file>         The path to directory to store files.
   --limit=<N>                  The number of files to read. -1 for all. [default: -1]
"""

import logging
logging.basicConfig(level=logging.INFO)
import numpy
import re
import os

import six
import sys
sys.path.insert(0, os.path.abspath('..'))

import utils

from collections import defaultdict
from nltk import pos_tag as pos_tagger
from nltk.parse.stanford import StanfordParser
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm


class Sentence(object):
    """Abstraction of a sentence."""
    def __init__(self, sentence_position, default_label='O'):
        # Position of the sentence in the document
        self.position = sentence_position
        self.words = []
        # Saves the start position in the original document for each word.
        self.word_positions = []
        self.pos = []
        self.labels = []
        self.tree = None
        self.default_label = default_label
        # A map from start positions (in document) to word index in self.words
        self.word_index = {}

    def add_word(self, word, pos, position, label=None):
        if not label:
            label = self.default_label
        self.words.append(word)
        self.pos.append(pos)
        self.labels.append(label)
        self.word_index[position] = len(self.words) - 1

    def build_from_text(self, text, initial_position=0):
        raw_words = word_tokenize(text)
        pos_tags = pos_tagger(raw_words)
        last_position = 0
        for index, (word, pos) in enumerate(pos_tags):
            last_position = text[last_position:].find(word)
            assert last_position >= 0
            self.add_word(word, pos, last_position + initial_position,
                          self.default_label)

    @property
    def start_position(self):
        if not self.words:
            return -1
        return self.words_positions[0]

    @property
    def end_position(self):
        if not self.words:
            return -1
        return self.words_positions[-1] + len(self.words[-1])

    def get_word_for_position(self, position):
        return self.word_index.get(position, -1)

    def add_label_for_position(self, start, end):
        """Adds label to words with positions between start and end (including).

        Returns the last position of char_range used. If char_range doesn't
        intersect with sentence, then does nothing."""
        start, end = char_range
        if end < self.start_position or start > self.end_position:
            return start



class EssayDocument(object):
    def __init__(self, identifier, default_label='O'):
        # List of Sentences.
        self.sentences = []
        # Representation of document as continous text.
        self.text = ''
        self.identifier = identifier
        self.default_label = default_label

    def build_from_text(self, text, start_index=0):
        self.text = text
        raw_sentences = sent_tokenize(six.text_type(text))
        for index, raw_sentence in enumerate(raw_sentence):
            sentence = Sentence(index)
            initial_position = text.find(sentence) + start_index
            assert initial_position >= 0
            sentence.build_from_text(raw_sentence, initial_position)
            self.sentences.append(sentence)

    def get_word_for_position(self, position):
        word_index = -1
        for sentence in self.sentences:
            if (position >= sentence.start_position and
                    position <= sentence.end_position):
                word_index = sentence.get_word_for_position(position)
                if word_index < 0:
                    raise IndexError('The index {} is not a word start'.format(
                        position))
                word_text = sentence.words[word_index]
                break
        assert text[position:len(word_text)] == word_text
        return word_index, word_text


    def add_label_for_position(self, label, char_range):
        """Adds the given label to all words covering range."""
        start, end = char_range
        assert start >= 0 and end < len(self.text)
        last_start = start
        for sentence in self.sentences:
            last_start = sentence.add_label_for_position(label, last_start, end)
            if last_start == end:
                break




class EssayDocumentFactory(object):
    """Builds a EssayDocument from input_file."""
    def __init__(self, input_filename):
        self.input_filename = input_filename
        self.label_input_file = None
        self.instance_input_file = None
        self.raw_labels = {}
        self.sentences = []
        self.title = ''
        self.parser = StanfordParser(
            model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',
            encoding='utf8')

    def __enter__(self):
        self.label_input_file = open(
            self.input_filename.replace('txt', 'ann'), 'r')
        self.instance_input_file = open(self.input_filename, 'r')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.label_input_file.close()
        self.instance_input_file.close()

    def get_labels(self):
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

    def _read_title(self):
        """Reads file first line as title."""
        self.title = self.instance_input_file.readline().strip()
            self.sentences = sent_tokenize(six.text_type(content.strip()))
            self.title = title.strip()

    def get_content(self):
        """Returns a tuple with the document title and body."""
        self._read_content()
        return self.title, self.sentences

    def get_parse_trees(self):
        """Returns the parse trees of the document content."""
        self._read_content()
        return self.parser.parse_sents(self.sentences)

    def build_document(self):
        """Creates a new EssayDocument instance."""
        self._read_title()
        for paragraph in self.instance_input_file.readline():
            if paragraph == '':
                continue



def get_input_files(input_dirpath, pattern, limit=-1):
    """Returns the names of the files in input_dirpath that matches pattern."""
    all_files = os.listdir(input_dirpath)
    if limit < 0:
        limit = len(all_files)
    result = []
    for filename in all_files:
        if re.match(pattern, filename) and os.path.isfile(os.path.join(
                input_dirpath, filename)):
            result.append(os.path.join(input_dirpath, filename))
        if len(result) > limit:
            break
    return result


def main():
    """Main fuction of the script."""
    args = utils.read_arguments(__doc__)
    documents = []
    labels = []
    filenames = get_input_files(args['input_dirpath'], r'.*txt',
                                int(args['limit']))
    for filename in tqdm(filenames):
        with EssayDocumentFactory(filename) as instance_extractor:
            title, content = instance_extractor.get_content()
            raw_labels = instance_extractor.get_labels()
            parse_tree = instance_extractor.get_parse_trees()
            document = EssayDocument(title)
            document.build(content, parse_tree, raw_labels)
            documents.append(document)

    # Convert labels to numeric vector
    unique_labels = sorted(numpy.unique(labels).tolist())
    y_vector = [unique_labels.index(label) for label in labels]
    logging.info('Classes used (sorted) {}'.format(unique_labels))

    utils.pickle_to_file((x_train, y_vector), args['output_filename'])



if __name__ == '__main__':
    main()
