# -*- coding: utf-8 -*-

"""Test for the AnnotatedDocument class."""

import unittest

from documents import AnnotatedDocument, AnnotatedJudgement


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

    def test_build_from_text(self):
        """Test if the document is built correctly."""
        self.document.build_from_text(self.text)
        self.assertEqual(self.text, self.document.text)
        for raw_sentence, sentence in zip(self.raw_sentences,
                                          self.document.sentences):
            self.assertEqual(raw_sentence, sentence.words)
            for index, word in enumerate(sentence.words):
                word_position = sentence.word_positions[index]
                self.assertEqual(
                    self.text[word_position:word_position+len(word)], word)

    def test_get_word(self):
        """Test the get_word_for_position function of a sentence"""
        self.document.build_from_text(self.text)
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
        self.document.build_from_text(self.text)
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
        self.document.build_from_text(self.text)
        self.document.add_label_for_position('A', 4, 13)
        expected_labels = ['O'] * len(self.raw_sentences[0])
        expected_labels[1] = 'A'
        expected_labels[2] = 'A'
        self.assertEqual(expected_labels, self.document.sentences[0].labels)

    def test_add_labels_second_sentence(self):
        self.document.build_from_text(self.text)
        self.document.add_label_for_position('A', 55, 72)
        expected_labels = ['O'] * len(self.raw_sentences[1])
        expected_labels[1] = 'A'
        expected_labels[2] = 'A'
        self.assertEqual(expected_labels, self.document.sentences[1].labels)

    def test_add_labels_multiple_sentence(self):
        self.document.build_from_text(self.text)
        self.document.add_label_for_position('A', 14, 72)
        expected_labels1 = ['A'] * len(self.raw_sentences[0])
        for index in range(0, 3):
            expected_labels1[index] = 'O'
        self.assertEqual(expected_labels1, self.document.sentences[0].labels)

        expected_labels2 = ['A'] * len(self.raw_sentences[1])
        expected_labels2[3] = 'O'
        expected_labels2[4] = 'O'
        self.assertEqual(expected_labels2, self.document.sentences[1].labels)


class AnnotatedJudgementTests(unittest.TestCase):
    """Test for the AnnotatedDocument class."""

    def test_build_from_large_text(self):
        """Test if the document is built correctly."""
        text = """SECOND SECTION
            CASE OF  ALKAŞI v. TURKEY
            (Application no. 21107/07)
            JUDGMENT
            STRASBOURG
            18 October 2016
            This judgment will become final in the circumstances set out in Article 44 § 2 of the Convention. It may be subject to editorial revision.
            In the case of Alkaşı v. Turkey,
            The European Court of Human Rights (Second Section), sitting as a Chamber composed of:
            Julia Laffranque,  President, Işıl Karakaş, Paul Lemmens, Valeriu Griţco, Ksenija Turković, Jon Fridrik Kjølbro, Georges Ravarani,  judges, and Hasan Bakırcı,  Deputy Section Registrar,
            Having deliberated in private on 27 September 2016,
            Delivers the following judgment, which was adopted on that date:
            PROCEDURE
            1.  The case originated in an application (no. 21107/07) against the Republic of Turkey lodged with the Court under Article 34 of the Convention for the Protection of Human Rights and Fundamental Freedoms (“the Convention”) by a Turkish national, Ms Ayten Alkaşı, on 9 May 2007.
            2.  The applicant was represented by Mr Hasan Gülşan, a lawyer practising in Istanbul. The Turkish Government (“the Government”) were represented by their Agent.
            3.  The applicant alleged that, despite her acquittal by a criminal court, the labour court’s subsequent judgment, and in particular the pronouncement of her guilt therein, had breached her right to be presumed innocent within the meaning of Article 6 § 2 of the Convention.
            4.  On 29 August 2013 the application was communicated to the Government.
            THE FACTS
        """
        document = AnnotatedJudgement('judg0')
        document.build_from_text(text)
        self.assertEqual(text, document.text)
        for sentence in document.sentences:
            for index, word in enumerate(sentence.words):
                word_position = sentence.word_positions[index]
                self.assertEqual(
                    text[word_position:word_position+len(word)], word)


if __name__ == '__main__':
    unittest.main()
