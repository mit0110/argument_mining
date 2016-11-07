"""Functions to convert conll formated EssayDocument into a numeric matrix."""

import logging

import os
import sys
import string
sys.path.insert(0, os.path.abspath('..'))
from collections import defaultdict
from tqdm import tqdm

import utils

PUNCTUATION_MARKS = set(string.punctuation)

class ConllFeatureExtractor(object):
    """Converts EssayDocument list into numeric matrix."""

    def __init__(self, use_structural=True):
        self.use_structural = use_structural

    def get_structural_features(self, document):
        """Adds structural features to the dictionary features."""
        if not self.use_structural:
            return
        word_position_in_paragraph = 0
        word_position_in_document = 0
        instances = []
        for sent_index, sentence in enumerate(document.sentences):
            if sentence.position_in_paragraph == 0:
                word_position_in_paragraph = 0
            for token_index, token, _ in sentence.iter_words():
                features = defaultdict(int)
                # Is the word in the introduction or the conclusion?
                features['st:tk:introduction'] = sentence.paragraph_number == 0
                features['st:tk:conclusion'] = (sentence.paragraph_number ==
                    document.sentences[-1].paragraph_number)
                # Position of word in sentence
                features['st:tk:position_in_sentence'] = token_index
                features['st:tk:first'] = token_index == 0
                features['st:tk:last'] = token_index == len(sentence) - 1
                # Position in paragraph
                features['st:tk:pos_in_paragraph'] = word_position_in_paragraph
                word_position_in_paragraph += 1
                # Position in document
                features['st:tk:pos_in_document'] = word_position_in_document
                word_position_in_document += 1

                # Punctuation marks
                features['st:pn:is_full_stop'] = (token == '.' and
                    token_index == len(sentence) - 1)
                features['st:pn:is_puntuation'] = token in PUNCTUATION_MARKS
                if token_index > 0:
                    prev_token = sentence.words[token_index - 1]
                elif sent_index > 0:
                    prev_token = document.sentences[sent_index - 1].words[-1]
                    features['st:pn:follows_full_stop'] = prev_token == '.'
                else:
                    prev_token = ''
                features['st:pn:follows'] = prev_token in PUNCTUATION_MARKS
                features['st:pn:follows_comma'] = prev_token == ','
                features['st:pn:follows_semicolon'] = prev_token == ';'
                if not (sent_index == 0 and token_index == 0):
                    # Is not the first word in the document
                    instances[-1]['st:pn:preceeds'] = token in PUNCTUATION_MARKS
                    instances[-1]['st:pn:preceeds_comma'] = token == ','
                    instances[-1]['st:pn:preceeds_semicolon'] = token == ';'
                    if features['st:tk:last']:
                        instances[-1]['st:pn:preceeds_full_stop'] = token == '.'

                # Position of covering sentence
                features['st:cs:position_in_paragraph'] = (
                    sentence.position_in_paragraph)
                features['st:cs:position_in_document'] = (
                    sentence.position_in_document)

                instances.append(features)
        return instances

    def get_feature_dict(self, documents):
        """Returns a dictionary of features."""
        instances = []
        for document in tqdm(documents):
            structural_features = self.get_structural_features(document)
            instances.extend(structural_features)
        return instances

    def transform(self, documents):
        """Returns a numpy array with extracted features."""
        instances = self.get_feature_dict(documents)
        # Convert to matrix and return
