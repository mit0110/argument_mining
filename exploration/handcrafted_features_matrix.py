"""Builds a matrix with the handcrafted features in preprocess.conll_feature_extractor.

Usage:
   handcrafted_features_matrix.py --input_filename=<filename>

Options:
    --input_filename=<filename>      The path to directory to read the dataset.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../experiments/'))

import conll_feature_extractor 
import utils


def main():
    """Main function of script"""
    args = utils.read_arguments(__doc__)
    print('Loading documents')
    documents = utils.pickle_from_file(args['input_filename'])

    transformer = conll_feature_extractor.ConllFeatureExtractor(
        use_structural=True,
        # use_syntactic=True,
        use_lexical=True
    )
    # Extract instances and labels. Each instance is a sentence, represented as
    # a list of feature dictionaries for each work. Labels are represented as
    # a list of word labels.
    instances = transformer.get_feature_dict(documents)
    labels = conll_feature_extractor.get_labels_from_documents(documents)

    print('All operations completed')


if __name__ == '__main__':
    main()

