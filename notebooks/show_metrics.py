"""Functions to print metrics associated to annotations"""
import matplotlib.pyplot as plt
import numpy
import os
import seaborn as sns

from sklearn import metrics
from read_annotations import append_path

THIRD_PARTY_DIR = '../../third_party/'


append_path(THIRD_PARTY_DIR)
append_path(os.path.join(THIRD_PARTY_DIR, 'krippendorff-alpha'))
import fleiss_kappa
from krippendorff_alpha import krippendorff_alpha, nominal_metric


def show_kappa(labels1, labels2, identifier1, identifier2):
    kappa = metrics.cohen_kappa_score(labels1, labels2)
    print('Kohen-\'s Kappa {} - {}: {}'.format(
        identifier1.split('-')[1].strip(), identifier2.split('-')[1].strip(),
        kappa))


def show_krippendorff_alpha(labels):
    alpha = krippendorff_alpha(labels, metric=nominal_metric,
                               convert_items=lambda x:x, missing_items=[])
    print('Krippendorff Alpha: {}'.format(alpha))


def show_fleiss_kappa(labels):
    input = []
    for label in labels:
        input.append(numpy.unique(label, return_counts=True)[1])
    kappa = fleiss_kappa.fleiss_kappa(numpy.array(input))
    print('Fleiss Kappa: {}'.format(kappa))


def show_confusion_matrix(labels1, labels2, identifier1=None, identifier2=None):
    label_names = sorted(list(set(labels1)))
    matrix = metrics.confusion_matrix(labels1, labels2, labels=label_names)
    observed_agreement = numpy.trace(matrix) / float(numpy.sum(matrix)) * 100
    print('Observed Agreement: {0:.2f}%'.format(observed_agreement))
    figure = sns.heatmap(matrix, annot=True, fmt="d", linewidths=.5,
                         xticklabels=label_names, yticklabels=label_names)
    if identifier1 is not None:
        figure.set(ylabel=identifier1)
    if identifier2 is not None:
        figure.set(xlabel=identifier2)
    plt.show()

