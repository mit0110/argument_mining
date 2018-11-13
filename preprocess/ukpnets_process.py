"""Script to preprocess the dataset for UKP-lstm networks.

We can't use the same script as it looks for data in a different directory.

The name of the dataset is the name of the folder where it is located
"""

from __future__ import absolute_import

import argparse
import pickle
import os
import utils

from ukplab_nets.util import preprocessing


def read_args():
    parser = argparse.ArgumentParser(
        description='Training a bi-directional RNN')
    # Classifier parameters
    parser.add_argument('--embeddings_path', type=str,
                        help='Path to embedding file')
    parser.add_argument('--output_dirpath', type=str,
                        help='Path to store the resulting pickled file')
    parser.add_argument('--dataset', type=str,
                        help='Path to directory with the dataset. Must be '
                             'in different files train.txt, test.txt, dev.txt')
    parser.add_argument('--name', type=str, default=None,
                        help='Name of the dataset to use. If None, then the '
                             'name of the directory will be used.')

    args = parser.parse_args()

    return args


def prepare_dataset(embeddings_path, datasets, output_dirpath,
                    freq_threshold_unk_tokens=50,
                    reduce_embeddings=False, value_transformations=None,
                    pad_onetoken_sentence=True):
    """Preprocess dataset and embeddings.

    Reads in the pre-trained embeddings (in text format) from embeddings_path
    and prepares those to be used with the LSTM network.
    Unknown words in the trainDataPath-file are added, if they appear at least
    freq_threshold_unk_tokens times

    Args:
        embeddings_path: Full path to the pre-trained embeddings file.
            File must be in text format.
        datasets: A dictionary where the keys are the dataset names and the
            values are the specification for the dataset. The specifications are
            also dicts, with the keys columns, labels, evaluate, commentSymbol
            and dirpath. dirpath contains the path to the directory where the
            three partitions of the dataset (train, test, dev) are stored in
            txt format.
        output_dirpath: Path to directory to store the resulting pickled file
        freq_threshold_unk_tokens: Unknown words are added, if they occure more
            than freq_threshold_unk_tokens times in the train set
        reduce_embeddings: Set to true, then only the embeddings needed for
            training will be loaded
        value_transformations: Column specific value transformations
        pad_onetoken_sentence: True to pad one sentence tokens
            (needed for CRF classifier)
    """
    utils.safe_mkdir(output_dirpath)
    embeddings_name = os.path.basename(embeddings_path)[:10]
    dataset_name = "_".join(sorted(datasets.keys()) + [embeddings_name])
    output_filename = os.path.join(output_dirpath, dataset_name + '.p')

    casing2Idx = preprocessing.getCasingVocab()
    embeddings, word2Idx = preprocessing.readEmbeddings(
        embeddings_path, datasets,
        freq_threshold_unk_tokens, reduce_embeddings)

    mappings = {'tokens': word2Idx, 'casing': casing2Idx}
    result = {'embeddings': embeddings, 'mappings': mappings,
              'datasets': datasets, 'data': {}}

    for name, dataset in datasets.items():
        trainData = os.path.join((dataset['dirpath']), 'train.txt')
        devData = os.path.join((dataset['dirpath']), 'dev.txt')
        testData = os.path.join((dataset['dirpath']), 'test.txt')
        paths = [trainData, devData, testData]
        print(paths)

        result['data'][name] = preprocessing.createPklFiles(
            paths, mappings, dataset['columns'], dataset['commentSymbol'],
            value_transformations, pad_onetoken_sentence)

    utils.pickle_to_file(result, output_filename)

    print("DONE - Embeddings file saved: {}".format(output_filename))


def main():
    args = read_args()
    # The name of the dataset is the name of the folder where it is located
    if args.name is None:
        dataset_name = os.path.split(os.path.split(args.dataset)[0])[1]
    else:
        dataset_name = args.name

    datasets = {
        dataset_name: { # Name of the dataset
            # Name of the columns
            'columns': {1: 'tokens', 2: 'arg_component'},
            # Directory of the dataset
            'dirpath': args.dataset,
            # Which column we want to predict
            'label': 'arg_component',
            # Should we evaluate on this task? Set true always for single task
            'evaluate': True,
            # Lines in the input data starting with this string will be skipped.
            'commentSymbol': None
        }
    }
    prepare_dataset(args.embeddings_path, datasets, args.output_dirpath,
                    reduce_embeddings=False, freq_threshold_unk_tokens=None)


if __name__ == '__main__':
    main()
