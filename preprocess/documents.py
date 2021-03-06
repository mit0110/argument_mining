"""Abstractions to handle AnnotatedDocuments"""
import re

from collections import defaultdict
from nltk import pos_tag as pos_tagger
from nltk.tokenize import sent_tokenize, word_tokenize
from termcolor import colored


class Sentence(object):
    """Abstraction of a sentence."""
    def __init__(self, position_in_document=-1, default_label='O',
                 paragraph_number=-1, position_in_paragraph=-1):
        # Position of the sentence in the document
        self.position_in_document = position_in_document
        self.paragraph_number = paragraph_number
        self.position_in_paragraph = position_in_paragraph
        self.words = []
        # Saves the start position in the original document for each word.
        self.word_positions = []
        self.pos = []
        self.labels = []
        self.attributes = []
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
        self.attributes.append(None)
        self.word_positions.append(position)
        self.word_index[position] = len(self.words) - 1

    def build_from_text(self, text, initial_position=0):
        raw_words = word_tokenize(text)
        pos_tags = pos_tagger(raw_words)
        last_position = 0
        for word, pos in pos_tags:
            word_position = text[last_position:].find(word) + last_position
            if word_position < last_position:
                # This happens because the word tokenizer changes the codes
                # for the quotes
                if word == u'``' or word == "''":
                    word_position = text[last_position:].find(
                        '"') + last_position
                    word = '"'
                else:
                    raise IndexError('Word {} not in sentence {}'.format(
                        word, text))
            self.add_word(word, pos, word_position + initial_position,
                          self.default_label)
            last_position = word_position + len(word)

    @property
    def start_position(self):
        if not self.words:
            return -1
        return self.word_positions[0]

    @property
    def end_position(self):
        if not self.words:
            return -1
        return self.word_positions[-1] + len(self.words[-1])

    def get_word_for_position(self, position):
        return self.word_index.get(position, -1)

    def add_label_for_position(self, label, start, end, attribute=None):
        """Labels words with positions between start and end (not included).

        Returns the last position of char_range used. If char_range doesn't
        intersect with sentence, then does nothing."""
        if end < self.start_position or start >= self.end_position:
            return start
        while start < end and start < self.end_position:
            word_index = self.get_word_for_position(start)
            if word_index < 0:
                word_index = 0
                for position1, position2 in zip(self.word_positions[:-1],
                                                self.word_positions[1:]):
                    if not(start <= position2 and start >= position1):
                        word_index += 1
                    else:
                        break
            self.labels[word_index] = label
            if attribute is not None:
                self.attributes[word_index] = attribute
            if word_index == len(self.words) - 1:  # end of sentence
                start += len(self.words[word_index]) + 1
            else:
                start = self.word_positions[word_index + 1]
        return start

    def iter_words(self):
        """Iterates over tuples of words (position in sentence, word, pos)"""
        for index, (token, tag) in enumerate(zip(self.words, self.pos)):
            yield index, token, tag

    @property
    def has_label(self):
        """Returns true if any word in the sentence is labeled with other than
        the default label."""
        return len(set(self.labels)) > 1 or self.labels[0] != self.default_label

    def pretty_print(self, styles=None):
        if styles is None:
            styles = defaultdict(lambda: 'blue')
        result = ''
        previous_label = self.default_label
        for word, label in zip(self.words, self.labels):
            if label != previous_label:
                if label != self.default_label:  # new block
                    result += '{'
                else:  # end of block
                    result += '}'
            if label == self.default_label:
                result += word + ' '
            else:
                result += colored(word, styles[label]) + ' '
            previous_label = label
        if previous_label != self.default_label:
            result += '}'
        return result

    def __len__(self):
        return len(self.words)

    def __repr__(self):
        return ' '.join(self.words)


class UnlabeledDocument(object):
    """A class to represent a processed unlabeled document of text."""

    def __init__(self, identifier, title=''):
        # List of Sentences.
        self.sentences = []
        # Representation of document as continous text.
        self.text = ''
        self.identifier = identifier
        self.title = title
        self.parse_trees = []

    def build_from_text(self, text, start_index=0):
        self.text = text
        if isinstance(text, str) or isinstance(text, unicode):
            paragraphs = self.text.split('\n')
        else:
            paragraphs = text
        position_in_document = 0
        seen_text = 0
        for paragraph_index, paragraph in enumerate(paragraphs):
            raw_sentences = sent_tokenize(paragraph)
            for index, raw_sentence in enumerate(raw_sentences):
                sentence = Sentence(position_in_document=position_in_document,
                                    paragraph_number=paragraph_index,
                                    position_in_paragraph=index)
                if isinstance(text, str):
                    initial_position = (self.text[seen_text:].find(raw_sentence) +
                                        start_index + seen_text)
                    assert initial_position >= 0
                else:
                    initial_position = 0
                seen_text = initial_position
                sentence.build_from_text(raw_sentence, initial_position)
                self.sentences.append(sentence)
                position_in_document += 1

    def parse_text(self, parser):
        # for sentence in self.sentences:
        try:
            self.parse_trees = list(parser.parse_sents(
                [sentence.words for sentence in self.sentences]))
        except UnicodeDecodeError:
            print('Error in parse tree {}'.format(self.identifier))
            self.parse_trees.append(None)

    def __repr__(self):
        return self.identifier


class AnnotatedDocument(UnlabeledDocument):

    def __init__(self, identifier, default_label='O', title=''):
        # List of Sentences.
        super(AnnotatedDocument, self).__init__(identifier, title)
        self.default_label = default_label
        self.annotated_components = {}
        self.named_components = {} # Temporal map to store the original names
        # of the components to read relations later.
        self.annotated_relations = defaultdict(dict)

    def add_label_for_position(self, label, start, end, attribute=None):
        """Adds the given label to all words covering range."""
        if not (start >= 0 and end <= self.sentences[-1].end_position):
            print('WARNING: attempting to set a label from position '
                  '{} to {} in document with max len of {}'.format(
                start, end, self.sentences[-1].end_position))
        last_start = start
        for sentence in self.sentences:
            if last_start > sentence.end_position:
                continue
            last_start = sentence.add_label_for_position(
                label, last_start, end, attribute=attribute)
            if last_start >= end - 1:
                break

    def add_component(self, name, start, end):
        self.annotated_components[start] = end
        self.named_components[name] = start

    def add_relation(self, label, arg1, arg2):
        start1 = self.named_components[arg1.strip()]
        start2 = self.named_components[arg2.strip()]
        self.annotated_relations[start1][start2] = label

    def get_relative_relations(self):
        """Returns self.annotated_relations in relative values"""
        result = defaultdict(dict)
        starts = sorted(self.annotated_components.keys())
        for start1, relation_dict in self.annotated_relations.items():
            for start2, label in relation_dict.items():
                index_start1 = starts.index(start1)
                index_start2 = starts.index(start2)
                relative_start2 = index_start2 - index_start1
                assert relative_start2 != 0
                result[start1][relative_start2] = label
        return result

    def sample_labeled_text(self, limit=10, styles=None):
        sample = ''
        for sentence in self.sentences[:limit]:
            if not sentence.has_label:
                continue
            sample += sentence.pretty_print(styles) + '\n\n'
        return sample

    def get_word_label_list(self):
        """Returns a tuple with the list of words and the list of labels"""
        words = [w for sentence in self.sentences for w in sentence.words]
        labels = [l for sentence in self.sentences for l in sentence.labels]
        return words, labels

    def has_annotation(self):
        """Returns True if the document has any non default label"""
        for sentence in self.sentences:
            if sentence.has_label:
                return True
        return False


class AnnotatedJudgement(AnnotatedDocument):
    """Represents annotated judgements, separating the different sections."""

    SECTION_REGEX = re.compile('^[A|B|C|D|III|II|I|IV]\.\s*(.{,50})$')

    def get_initial_position(self, previous_sentence, raw_sentence,
                             start_index, last_sentence_start):
        if previous_sentence != '':
            initial_position = self.text.find(
                previous_sentence, last_sentence_start) + start_index
            current_initial_position = self.text.find(
                raw_sentence, last_sentence_start) + start_index
            missing_spaces = (current_initial_position -
                              initial_position - len(previous_sentence))
            raw_sentence = (previous_sentence + ' ' * missing_spaces +
                            raw_sentence)
            previous_sentence = ''
        else:
            initial_position = self.text.find(
                raw_sentence, last_sentence_start) + start_index
        return initial_position, previous_sentence, raw_sentence

    def build_from_text(self, text, start_index=0):
        try:
            self.text = text.decode('utf-8')
        except AttributeError:
            self.text = text
        current_section = 'Introduction'
        paragraphs = self.text.split('\n')
        position_in_document = 0
        last_sentence_start = start_index
        for paragraph_index, paragraph in enumerate(paragraphs):
            raw_sentences = sent_tokenize(paragraph)
            index = 0
            previous_sentence = ''
            for raw_sentence in raw_sentences:
                # Fix the tokenization for line numbers
                if len(raw_sentence) < 4:
                    previous_sentence = raw_sentence
                    continue
                initial_position, previous_sentence, raw_sentence = (
                    self.get_initial_position(previous_sentence,
                                              raw_sentence, start_index,
                                              last_sentence_start)
                )
                assert initial_position >= 0
                candidate_section = self.SECTION_REGEX.search(raw_sentence)
                if candidate_section is not None:
                    current_section = candidate_section.group(1).strip()
                sentence = Sentence(position_in_document=position_in_document,
                                    paragraph_number=paragraph_index,
                                    position_in_paragraph=index)
                sentence.section = current_section
                sentence.build_from_text(raw_sentence, initial_position)
                self.sentences.append(sentence)
                position_in_document += 1
                index += 1
                last_sentence_start += len(raw_sentence)

