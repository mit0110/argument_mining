"""Functions to print metrics associated to annotations"""
import matplotlib.pyplot as plt
import numpy
import os
import seaborn as sns

from sklearn import metrics
from read_annotations import append_path, get_labels

THIRD_PARTY_DIR = '../../third_party/'

try:
    append_path(THIRD_PARTY_DIR)
    append_path(os.path.join(THIRD_PARTY_DIR, 'krippendorff-alpha'))
    import fleiss_kappa
    from krippendorff_alpha import krippendorff_alpha, nominal_metric
except:
    fleiss_kappa = None
    krippendorff_alpha = None


def show_kappa(labels1, labels2, identifier1, identifier2):
    kappa = metrics.cohen_kappa_score(labels1, labels2)
    print('Kohen-\'s Kappa {} - {}: {}'.format(
        identifier1.split('-')[1].strip(), identifier2.split('-')[1].strip(),
        kappa))


def show_krippendorff_alpha(labels):
    alpha = None
    if krippendorff_alpha is not None:
        alpha = krippendorff_alpha(labels, metric=nominal_metric,
                                   convert_items=lambda x:x, missing_items=[])
        print('Krippendorff Alpha: {0:.3f}'.format(alpha))
    else:
        print('No module Krippendorff Alpha')
    return alpha


def show_fleiss_kappa(labels):
    kappa = None
    if fleiss_kappa is not None:
        input = []
        for label in labels:
            input.append(numpy.unique(label, return_counts=True)[1])
        kappa = fleiss_kappa.fleiss_kappa(numpy.array(input))
        print('Fleiss Kappa: {0:.3f}'.format(kappa))
    else:
        print('No module Fleiss Kappa')
    return kappa


def show_confusion_matrix(labels1, labels2, identifier1=None, identifier2=None):
    label_names = sorted(list(set(labels1)))
    matrix = metrics.confusion_matrix(labels1, labels2, labels=label_names)
    observed_agreement = numpy.trace(matrix) / float(numpy.sum(matrix)) * 100
    print('Observed Agreement: {0:.2f}%'.format(observed_agreement))
    colormap = plt.cm.cubehelix_r
    figure = sns.heatmap(matrix, annot=True, fmt="d", linewidths=.5,
                         xticklabels=label_names, yticklabels=label_names,
                         cmap=sns.cubehelix_palette(8,  as_cmap=True))
    if identifier1 is not None:
        figure.set(ylabel=identifier1)
    if identifier2 is not None:
        figure.set(xlabel=identifier2)
    plt.show()


def get_annotator(document):
    return document.identifier.split('-')[1].split(':')[1].strip()


def show_general_agreement(document_pairs, process_function=None, annotators=2):
    seen_annotators = []
    def get_annotator_index(document):
        annotator = get_annotator(document)
        if annotator not in seen_annotators:
            seen_annotators.append(annotator)
            return len(seen_annotators) - 1
        return seen_annotators.index(annotator)

    result = numpy.full((annotators, annotators), numpy.nan)
    for doc1, doc2 in document_pairs:
        ann1_index = get_annotator_index(doc1)
        ann2_index = get_annotator_index(doc2)
        labels1, labels2 = get_labels(doc1, doc2)
        if process_function is not None:
            labels1 = process_function(labels1)
            labels2 = process_function(labels2)
        kappa = metrics.cohen_kappa_score(labels1, labels2)
        # Todo change this to support multiple documents
        result[ann1_index, ann2_index] = kappa
        result[ann2_index, ann1_index] = kappa
    figure = sns.heatmap(result, annot=True, fmt=".2f", linewidths=.5,
                         vmax=1, xticklabels=seen_annotators,
                         yticklabels=seen_annotators)
    plt.title('Cohen\'s Kappa agreement between annotators')
    plt.show()
