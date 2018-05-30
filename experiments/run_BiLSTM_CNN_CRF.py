"""Script to use a trained BiLSTM-CNN-CRF model to tag a new document.

To run the script, clone the repository
https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf.git
under the name ukplab_nets and add it the path to PYTHONPATH.
"""

import argparse
import os
import numpy
import pandas
import utils

from sklearn import metrics
from ukplab_nets.neuralnets.BiLSTM import BiLSTM


def read_args():
    parser = argparse.ArgumentParser(
        description='Loading a bi-directional RNN')
    # Classifier parameters
    parser.add_argument('--classifier', type=str, default='CRF',
                        help='Path to the .h5 file with the classifier.')
    parser.add_argument('--dataset', type=str,
                        help='Path to the pickled file with the dataset')
    parser.add_argument('--output_dirname', type=str,
                        help='Path to store the predictions for dev '
                             'and test datasets')
    parser.add_argument('--target_column', type=str, default='arg_component',
                        help='Name of the column to use as label.')
    args = parser.parse_args()

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

    dataset_name = [x for x in data.keys()][0]  # I hate python 3
    label_encoding = {value: key
                      for key, value in mappings[args.target_column].items()}

    model = BiLSTM.loadModel(args.classifier)

    def tag_dataset(partition_name):
        tags = model.tagSentences(data[dataset_name][partition_name])
        assert (len(data[dataset_name][partition_name][0]['raw_tokens']) ==
                len(tags[dataset_name][0]))
        true_labels = []
        result = []

        for sentence, sentence_labels in zip(data[dataset_name][partition_name],
                                             tags[dataset_name]):
            for token, true_label_id, predicted_label in zip(
                    sentence['raw_tokens'], sentence[args.target_column],
                    sentence_labels):
                true_label = label_encoding[true_label_id]
                true_labels.append(true_label)
                result.append((token, true_label, predicted_label))

        result = pandas.DataFrame(result)
        partition_name_short = 'dev' if 'dev' in partition_name else 'test'
        output_filename = os.path.join(
            args.output_dirname,
            'predictions_{}_{}.conll'.format(dataset_name, partition_name_short))
        result.to_csv(output_filename, sep='\t', index=False)
        print(metrics.classification_report(
    	    true_labels, numpy.concatenate(tags[dataset_name])))

    tag_dataset('devMatrix')
    tag_dataset('testMatrix')


if __name__ == '__main__':
    main()
