"""Script to train a BiLSTM-CNN-CRF model from UKPLab repository

To run the script, clone the repository
https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf.git
under the name ukplab_nets and add it the path to PYTHONPATH.
"""

import argparse
import logging
import os
import sys
import utils

from ukplab_nets.neuralnets.BiLSTM import BiLSTM


loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def read_args():
    parser = argparse.ArgumentParser(
        description='Training a bi-directional RNN')
    # Classifier parameters
    parser.add_argument('--num_units', nargs='+', default=[100, 100], type=int,
                        help='Number of hidden units in RNN')
    parser.add_argument('--dropout', nargs='+', default=[0.5, 0.5], type=float,
                        help='Dropout ratio for every layer')
    parser.add_argument('--char_embedding', type=str, default=None,
                        choices=[None, 'lstm', 'cnn'],
                        help='Type of character embedding. Options are: None, '
                        'lstm or cnn. LSTM embeddings are from '
                        'Lample et al., 2016, CNN embeddings from '
                        'Ma and Hovy, 2016')
    parser.add_argument('--char_embedding_size', type=int, default=30,
                        help='Size of the character embedding. Use 0 '
                        'for no embedding')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of sentences in each batch')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of iterations of lower results before '
                        'early stopping.')
    # TODO add options for char embedding sizes
    # TODO add options for clipvalue and clipnorm

    # Pipeline parametres
    parser.add_argument('--dataset', type=str,
                        help='Path to the pickled file with the dataset')
    parser.add_argument('--output_dirpath', type=str,
                        help='Path to store the performance scores for dev '
                             'and test datasets')
    parser.add_argument('--output_prediction', action='store_true',
                        help='Output predictions to temp files')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train the classifier')
    parser.add_argument('--classifier', type=str, default='CRF',
                        help='Classifier type (last layer). Options are '
                             'CRF or Softmax.')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment to store the results')
    args = parser.parse_args()

    assert len(args.num_units) == len(args.dropout)
    return args


def load_dataset(filename):
    pickled_object = utils.pickle_from_file(filename)
    return (pickled_object['embeddings'], pickled_object['mappings'],
            pickled_object['data'], pickled_object['datasets'])


def main():
    """Training pipeline"""
    args = read_args()

    # Read dataset
    embeddings, mappings, data, datasets = load_dataset(args.dataset)

    classifier_params = {
        'classifier': args.classifier, 'LSTM-Size': args.num_units,
        'dropout': args.dropout, 'charEmbeddingsSize': args.char_embedding_size,
        'charEmbeddings': args.char_embedding, 'miniBatchSize': args.batch_size,
        'earlyStopping': args.patience
    }
    print(classifier_params)

    model = BiLSTM(classifier_params)
    model.setMappings(mappings, embeddings)
    model.setDataset(datasets, data)
    # Path to store performance scores for dev / test
    if args.experiment_name is None:
        name = os.path.join(
            args.output_dirpath,
            '_'.join([args.classifier, str(args.char_embedding)] +
                     [str(x) for x in args.num_units])
        )
    else:
        name = os.path.join(args.output_dirpath, args.experiment_name)
    model.storeResults(name)
    # Path to store models
    model.modelSavePath = os.path.join(
        args.output_dirpath, "[ModelName]_[DevScore]_[TestScore]_[Epoch].h5")
    model.fit(epochs=args.epochs)


if __name__ == '__main__':
    main()