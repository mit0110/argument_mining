"""Test for the ConllFeatureExtractor class."""

import unittest

import os
import sys
# To unpickle the test document
sys.path.insert(0, os.path.abspath('../preprocess/'))
sys.path.insert(0, os.path.abspath('../'))

import utils
import annotated_documents
from conll_feature_extractor import ConllFeatureExtractor, get_parent_sibling
from nltk.tree import Tree
from scipy.sparse import csr_matrix


class TestConllFeaturesExtractor(unittest.TestCase):

    DOCUMENT = (u"Nowadays, the importance of vehicles has been widely "
        "recognized thoughout the world. However, it rises concerns about the "
        "increasingly severe traffic problems as well as the automobile exhaust"
        " pollution. Some argue that the best solution to these issues is "
        "raising the price of petrol. From my perspective, I think their view "
        "is overly simplistic.\nUndeniably, making the fuel cost more money "
        "could limit the number of vehicles to some extent. Due to the "
        "increasing number of petrol stations, the competition in this field "
        "is more and more fierce, thus the price of petrol could be lower in "
        "the future. Therefore, those who are suffering from the inefficiency "
        "of public transport tend to own a car immediately. Moreover, some "
        "families purchase even more than three cars just for their own sake. "
        "In such case if they must pay more money on fuel, they may consider "
        "about that price of purchasing additionally for more time.\n")
    LABELS = [
        ('Claim', 384, 415), ('Claim', 429, 510), ('Premise', 512, 669),
        ('Premise', 682, 777), ('Premise', 789, 861), ('Premise', 876, 988),
    ]
    ORIGINAL_TITLE = ('Can petrol price increase impact on reducing traffic '
        'and pollution?\n\n')

    def setUp(self):
        self.document = annotated_documents.AnnotatedDocument('testDoc',
                                                      title=self.ORIGINAL_TITLE)
        self.document.build_from_text(self.DOCUMENT,
                                      start_index=len(self.ORIGINAL_TITLE))
        for label, start_index, end_index in self.LABELS:
            self.document.add_label_for_position(label, start_index, end_index)

    def test_instances(self):
        """Test the number of obtained instances"""
        extractor = ConllFeatureExtractor(use_structural=True)
        instances = extractor.get_feature_dict([self.document])
        self.assertEqual(
            sum([len(sentence.words) for sentence in self.document.sentences]),
            sum([len(sentence_features) for sentence_features in instances]))

    def test_structural_features(self):
        extractor = ConllFeatureExtractor(use_structural=True)
        # Sentence 3 word 0, From
        features = extractor.get_structural_features(self.document)[3][0]
        # Token position features
        self.assertEqual(True, features['st:tk:introduction'])
        self.assertEqual(False, features['st:tk:conclusion'])
        # Position in covering sentence.
        self.assertEqual(0, features['st:tk:position_in_sentence'])
        self.assertEqual(True, features['st:tk:first'])
        self.assertEqual(False, features['st:tk:last'])
        self.assertEqual(49, features['st:tk:pos_in_paragraph'])
        self.assertEqual(49, features['st:tk:pos_in_document'])
        # Punctuation features
        self.assertEqual(False, features['st:pn:preceeds'])
        self.assertEqual(True, features['st:pn:follows'])
        self.assertEqual(False, features['st:pn:follows_comma'])
        self.assertEqual(False, features['st:pn:follows_semicolon'])
        self.assertEqual(True, features['st:pn:follows_full_stop'])
        self.assertEqual(False, features['st:pn:preceeds_comma'])
        self.assertEqual(False, features['st:pn:preceeds_semicolon'])
        self.assertEqual(False, features['st:pn:preceeds_full_stop'])
        self.assertEqual(False, features['st:pn:is_full_stop'])
        self.assertEqual(False, features['st:pn:is_puntuation'])
        # Position of covering sentence
        self.assertEqual(3, features['st:cs:position_in_paragraph'])
        self.assertEqual(3, features['st:cs:position_in_document'])

    def test_punctuation_first_word(self):
        """Test the puntuation features of the first word of the document"""
        extractor = ConllFeatureExtractor(use_structural=True)
        # Sentence 3 word 0, From
        features = extractor.get_structural_features(self.document)[0][0]
        # Punctuation features
        self.assertEqual(True, features['st:pn:preceeds'])
        self.assertEqual(False, features['st:pn:follows'])
        self.assertEqual(False, features['st:pn:follows_comma'])
        self.assertEqual(False, features['st:pn:follows_semicolon'])
        self.assertEqual(False, features['st:pn:follows_full_stop'])
        self.assertEqual(True, features['st:pn:preceeds_comma'])
        self.assertEqual(False, features['st:pn:preceeds_semicolon'])
        self.assertEqual(False, features['st:pn:preceeds_full_stop'])
        self.assertEqual(False, features['st:pn:is_full_stop'])
        self.assertEqual(False, features['st:pn:is_puntuation'])

    def test_syntactic_features(self):
        """Test extraction of syntactic features."""
        extractor = ConllFeatureExtractor(use_structural=False,
                                          use_syntactic=True)
        self.document.parse_trees = utils.pickle_from_file(
            os.path.join('test_files', 'parse_trees.pickle'))
        features = extractor.get_syntactic_features(self.document)[3][0]
        self.assertEqual('IN', features['syn:pos'])
        self.assertEqual(2, features['syn:lca:next'])
        self.assertEqual('PP', features['syn:lca:next_tag'])
        self.assertEqual(10, features['syn:lca:prev'])  # Height of the tree + 1
        self.assertEqual('', features['syn:lca:prev_tag'])

    def test_lexical_features(self):
        """Test extraction of syntactic features."""
        extractor = ConllFeatureExtractor(use_structural=False,
                                          use_lexical=True)
        self.document.parse_trees = utils.pickle_from_file(
            os.path.join('test_files', 'parse_trees.pickle'))
        features = extractor.get_lexical_features(self.document)[3][0]
        self.assertEqual('PP[From/IN]', features['ls:token_comb'])
        self.assertEqual('IN[From/IN]-NP[perspective/NN]',
                         features['ls:right_comb'])

    def test_get_parent_siblings(self):
        """Test the function get_parent_sibling."""
        tree = utils.pickle_from_file(
            os.path.join('test_files', 'parse_trees.pickle'))[0]
        expected_pairs = (
            (u',[,/,]', u'NP[importance/NN]'),
            (u'DT[the/DT]', u'NN[importance/NN]'),
            (u'IN[of/IN]', u'NP[vehicles/NNS]'),
            (u'VBZ[has/VBZ]', u'VP[been/VBN]'),
            (u'VBN[been/VBN]', u'ADVP[widely/RB]'),
            (u'VBN[recognized/VBN]', u'NP[thoughout/NN]'),
            (u'DT[the/DT]', u'NN[world/NN]')
        )
        result = []
        for leaf_index in range(len(tree.leaves())):
            pair = get_parent_sibling(tree, leaf_index)
            if pair:
                result.append(pair)
        for expected_pair, resulting_pair in zip(expected_pairs, result):
            self.assertEqual(expected_pair, resulting_pair)

    def test_matrix(self):
        """Test the size of the sparse matrix generated."""
        extractor = ConllFeatureExtractor(use_structural=True)
        self.document.parse_trees = utils.pickle_from_file(
            os.path.join('test_files', 'parse_trees.pickle'))
        matrix = extractor.transform([self.document])
        self.assertIsInstance(matrix, csr_matrix)
        self.assertEqual(
            sum([len(sentence.words) for sentence in self.document.sentences]),
            matrix.shape[0])


if __name__ == '__main__':
    unittest.main()
