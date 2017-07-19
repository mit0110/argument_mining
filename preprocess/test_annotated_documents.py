# -*- coding: utf-8 -*-

"""Test for the AnnotatedDocument class."""

import unittest

from annotated_documents import AnnotatedDocument


class AnnotatedDocumentTest(unittest.TestCase):
    """Test for the AnnotatedDocument class."""

    def setUp(self):
        self.text = '\n'.join([
            u'The quick brown fox jumps over them lazy dog.',
            u'¡Cuántas letras especiales añadidas!'
        ])
        self.raw_sentences = [
            u'The quick brown fox jumps over them lazy dog .'.split(),
            u'¡Cuántas letras especiales añadidas !'.split()
        ]
        self.document = AnnotatedDocument('doc0')
        self.document.build_from_text(self.text)

    def test_build_from_text(self):
        """Test if the document is built correctly."""

        self.assertEqual(self.text, self.document.text)
        for raw_sentence, sentence in zip(self.raw_sentences,
                                          self.document.sentences):
            self.assertEqual(raw_sentence, sentence.words)

    def test_get_word(self):
        """Test the get_word_for_position function of a sentence"""
        for raw_sentence, sentence in zip(self.raw_sentences,
                                          self.document.sentences):
            for word in raw_sentence:
                # Works only because there are no repeated words.
                word_index = sentence.get_word_for_position(
                    self.text.find(word))
                self.assertNotEqual(-1, word_index)
                self.assertEqual(word, sentence.words[word_index])

    def test_get_word_with_start(self):
        """Test the get_word_for_position function with a start index."""
        self.document = AnnotatedDocument('doc0')
        self.document.build_from_text(self.text, start_index=10)
        for raw_sentence, sentence in zip(self.raw_sentences,
                                          self.document.sentences):
            for word in raw_sentence:
                # Works only because there are no repeated words.
                word_index = sentence.get_word_for_position(
                    self.text.find(word) + 10)
                self.assertNotEqual(-1, word_index)
                self.assertEqual(word, sentence.words[word_index])

    def test_add_labels(self):
        self.document.add_label_for_position('A', 4, 13)
        expected_labels = ['O'] * len(self.raw_sentences[0])
        expected_labels[1] = 'A'
        expected_labels[2] = 'A'
        self.assertEqual(expected_labels, self.document.sentences[0].labels)

    def test_add_labels_second_sentence(self):
        self.document.add_label_for_position('A', 55, 72)
        expected_labels = ['O'] * len(self.raw_sentences[1])
        expected_labels[1] = 'A'
        expected_labels[2] = 'A'
        self.assertEqual(expected_labels, self.document.sentences[1].labels)

    def test_add_labels_multiple_sentence(self):
        self.document.add_label_for_position('A', 14, 72)
        expected_labels1 = ['A'] * len(self.raw_sentences[0])
        for index in range(0, 3):
            expected_labels1[index] = 'O'
        self.assertEqual(expected_labels1, self.document.sentences[0].labels)

        expected_labels2 = ['A'] * len(self.raw_sentences[1])
        expected_labels2[3] = 'O'
        expected_labels2[4] = 'O'
        self.assertEqual(expected_labels2, self.document.sentences[1].labels)


if __name__ == '__main__':
    unittest.main()
