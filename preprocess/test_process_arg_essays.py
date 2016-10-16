"""Test for FeatureExtractor class"""

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


if __name__ == '__main__':
    unittest.main()
