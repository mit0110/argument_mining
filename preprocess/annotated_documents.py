"""Abstractions to handle AnnotatedDocuments"""
from collections import defaultdict

from nltk import pos_tag as pos_tagger
from nltk.tokenize import sent_tokenize, word_tokenize


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
        self.word_positions.append(position)
        self.word_index[position] = len(self.words) - 1

    def build_from_text(self, text, initial_position=0):
        raw_words = word_tokenize(text)
        pos_tags = pos_tagger(raw_words)
        last_position = 0
        for word, pos in pos_tags:
            word_position = text[last_position:].find(word) + last_position
            if word_position < last_position:
                if word == u'``':
                    word_position = text[last_position:].find(
                        '"') + last_position
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

    def add_label_for_position(self, label, start, end):
        """Adds label to words with positions between start and end (including).

        Returns the last position of char_range used. If char_range doesn't
        intersect with sentence, then does nothing."""
        if end < self.start_position or start > self.end_position:
            return start
        while start <= end and start < self.end_position:
            word_index = self.get_word_for_position(start)
            if word_index < 0:
                start += 1
                continue
            self.labels[word_index] = label
            start += len(self.words[word_index])
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
            styles = defaultdict(lambda: '**')  # boldtext
        result = ''
        for word, label in zip(self.words, self.labels):
            if label == self.default_label:
                result += word + ' '
            else:
                result += styles[label] + word + styles[label] + ' '
        return result

    def __len__(self):
        return len(self.words)

    def __repr__(self):
        return ' '.join(self.words)


class AnnotatedDocument(object):
    def __init__(self, identifier, default_label='O', title=''):
        # List of Sentences.
        self.sentences = []
        # Representation of document as continous text.
        self.text = ''
        self.identifier = identifier
        self.default_label = default_label
        self.title = title
        self.parse_trees = []
        self.annotated_components = {}
        self.named_components = {} # Temporal map to store the original names
        # of the components to read relations later.
        self.annotated_relations = defaultdict(dict)

    def build_from_text(self, text, start_index=0):
        self.text = text
        paragraphs = self.text.split('\n')
        position_in_document = 0
        for paragraph_index, paragraph in enumerate(paragraphs):
            raw_sentences = sent_tokenize(paragraph)
            for index, raw_sentence in enumerate(raw_sentences):
                sentence = Sentence(position_in_document=position_in_document,
                                    paragraph_number=paragraph_index,
                                    position_in_paragraph=index)
                initial_position = self.text.find(raw_sentence) + start_index
                assert initial_position >= 0
                sentence.build_from_text(raw_sentence, initial_position)
                self.sentences.append(sentence)
                position_in_document += 1

    def add_label_for_position(self, label, start, end):
        """Adds the given label to all words covering range."""
        assert start >= 0 and end <= self.sentences[-1].end_position
        last_start = start
        for sentence in self.sentences:
            last_start = sentence.add_label_for_position(label, last_start, end)
            if last_start == end:
                break

    def add_component(self, name, start, end):
        self.annotated_components[start] = end
        self.named_components[name] = start

    def add_relation(self, label, arg1, arg2):
        start1 = self.named_components[arg1]
        start2 = self.named_components[arg2]
        self.annotated_relations[start1][start2] = label

    def parse_text(self, parser):
        for sentence in self.sentences:
            self.parse_trees.append(next(parser.parse(sentence.words)))

    def __repr__(self):
        return self.identifier

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

