"""Test for FeatureExtractor class"""

import io
import unittest


import process_arg_essays


class TestFeatureExtractor(unittest.TestCase):
    """Test for the FeatureExtractor class"""

    def test_ngrams(self):
        """Test the generation of ngrams"""
        words = 'The cat sat on the mat. The cat sat on the corner.'.split()
        expected_counts = {word: 2 for word in words}
        expected_counts['mat.'] = 1
        expected_counts['corner.'] = 1
        expected_counts['The cat'] = 2
        expected_counts['cat sat'] = 2
        expected_counts['sat on'] = 2
        expected_counts['on the'] = 2
        expected_counts['the mat.'] = 1
        expected_counts['the corner.'] = 1
        expected_counts['mat. The'] = 1
        extractor = process_arg_essays.FeatureExtractor(ngrams_degree=2)
        self.assertEqual(expected_counts, extractor.count_ngrams(words))


class TestLabeledSentencesExtractor(unittest.TestCase):
    """Test for the LabeledSentencesExtractor class."""

    DOCUMENT = (u"Can petrol price increase impact on reducing traffic and "
        "pollution?\n\nNowadays, the importance of vehicles has been widely "
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
    LABELS = (u"T2\tClaim 384 415\ttheir view is overly simplistic\nA1\tStance "
        "T2 For\nT3\tClaim 429 510\tmaking the fuel cost more money could limit"
        " the number of vehicles to some extent\nT4\tPremise 512 669\tDue to "
        "the increasing number of petrol stations, the competition in this "
        "field is more and more fierce, thus the price of petrol could be lower"
        " in the future\nT6\tPremise 682 777\tthose who are suffering from the "
        "inefficiency of public transport tend to own a car immediately\nT7\t"
        "Premise 789 861\tsome families purchase even more than three cars just"
        " for their own sake\nT8\tPremise 876 988\tif they must pay more money "
        "on fuel, they may consider about that price of purchasing additionally"
        " for more time\n")

    EXPECTED_MATRIX = [
        [  # First paragraph
            'Nowadays, the importance of vehicles has been widely recognized '
            'thoughout the world.',
            'However, it rises concerns about the increasingly severe traffic '
            'problems as well as the automobile exhaust pollution.',
            'Some argue that the best solution to these issues is raising the '
            'price of petrol.',
            'From my perspective, I think their view is overly simplistic.'
        ],
        [  # Second paragraph
            'Undeniably, making the fuel cost more money could limit the number'
            ' of vehicles to some extent.',
            'Due to the increasing number of petrol stations, the competition '
            'in this field is more and more fierce, thus the price of petrol '
            'could be lower in the future.',
            'Therefore, those who are suffering from the inefficiency of public'
            ' transport tend to own a car immediately.',
            'Moreover, some families purchase even more than three cars just '
            'for their own sake.',
            'In such case if they must pay more money on fuel, they may '
            'consider about that price of purchasing additionally for more '
            'time.'
        ]
    ]

    EXPECTED_LABELS = [
        ['None', 'None', 'None', 'Claim'],
        ['Claim', 'Premise', 'Premise', 'Premise', 'Premise']
    ]

    def setUp(self):
        self.extractor = process_arg_essays.LabeledSentencesExtractor('Dummy')
        self.extractor.label_input_file = io.StringIO(self.LABELS)
        self.extractor.instance_input_file = io.StringIO(self.DOCUMENT)

    def test_read_labels(self):
        self.extractor._get_labels()
        expected_keys = {
            384: 'Claim', 429: 'Claim', 512: 'Premise',
            682: 'Premise', 789: 'Premise', 876: 'Premise'
        }
        self.assertEqual(sorted(expected_keys.keys()),
                         sorted(self.extractor.raw_labels.keys()))
        for key, label in expected_keys.iteritems():
            self.assertEqual(label, self.extractor.raw_labels[key][0])

    def test_get_label_for_sentence(self):
        self.extractor._get_labels()
        sentence = ('Therefore, those who are suffering from the inefficiency '
            'of public transport tend to own a car immediately.')
        expected_label = 'Premise'
        start_index = self.DOCUMENT.find(sentence)
        result = self.extractor._get_label(sentence, start_index)
        self.assertEqual(expected_label, result)

    def test_get_label_start(self):
        self.extractor._get_labels()
        self.assertEqual(-1, self.extractor.label_start_position(310, 312))
        self.assertEqual(384, self.extractor.label_start_position(380, 390))
        self.assertEqual(789, self.extractor.label_start_position(780, 890))
        self.assertEqual(-1, self.extractor.label_start_position(1000, 1890))

    def test_get_labeled_sentences(self):
        sentences, labels = self.extractor.get_labeled_sentences()
        self.assertEqual(self.EXPECTED_MATRIX, sentences)
        self.assertEqual(self.EXPECTED_LABELS, labels)


if __name__ == '__main__':
    unittest.main()
