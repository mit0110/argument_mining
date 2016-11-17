"""Base preprocess pipeline common for all classifiers."""

import itertools
import nltk
import numpy
import string

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer


def pos_filter(text):
    """Returns a string with verbs and adverbs."""
    def get_pos(sentence):
        words = nltk.tokenize.word_tokenize(sentence[0])
        return ''.join(['[[{}-{}]]'.format(word, tag)
                        for word, tag in nltk.pos_tag(words)
                        if tag.startswith('V') or tag in ['MD', 'RB']])
    result = numpy.apply_along_axis(get_pos, axis=0, arr=text)
    return result


def get_model_verbs(pos_tag_text):
    """Returns a column vector indicating if the instance contains a MD postag.
    """
    def has_modal(sentence):
        return int(sentence[0].find('MD]]') != -1)
    result = numpy.apply_along_axis(has_modal, axis=0, arr=pos_tag_text)
    return csr_matrix(result).T


def get_pos_steps():
    return [
        # Word features
        ('vect', CountVectorizer(ngram_range=(1, 3), max_features=10**4)),
        # Pos tag features
        ('pos', Pipeline([
            ('pos_extractor', FunctionTransformer(pos_filter)),
            ('pos_vectorizers', FeatureUnion([
                ('pos_vect', CountVectorizer(
                    ngram_range=(1, 1), max_features=1000,
                    token_pattern=u'\[\[.*?\]\]')),
                ('modal_extractor', FunctionTransformer(get_model_verbs)),
            ]))
        ]))
    ]


def get_basic_pipeline(classifier_tuple):
    """Returns an instance of sklearn Pipeline with the preprocess steps."""
    features = FeatureUnion(get_pos_steps())
    return Pipeline([
        ('features', features),
        ('tfidf', TfidfTransformer()),
        classifier_tuple
    ])


def get_basic_parameters():
    """Returns the possible parameters and values for the basic pipeline."""
    pos_features = get_pos_steps()
    return [{  # With pos
        'features__vect__max_features': [10**3, 10**4, 10**5],
        'features__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__use_idf': (True, False)
    },
    # {  # Without pos tags
    #     'features__vect__max_features': [10**3, 10**4, 10**5],
    #     'features__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    #     'features__transformer_list': pos_features[:1],
    #     'tfidf__use_idf': (True, False)
    # }
    ]


####


def get_words_from_tree(matrix):
    """Returns a string with the words from the tree."""
    result = []
    for document_trees in matrix:
        for paragraph_trees in document_trees:
            for sentence_tree in paragraph_trees:
                result.append(' '.join(sentence_tree[0].leaves()))
    return numpy.array(result)


def get_pos_from_tree(matrix):
    """Returns a string with PoS tags from the tree"""
    def get_filtered_pos(sentence_tree):
        """Returns verb, adverb (pos tags and word) and modal (pos tags)"""
        result = ''
        has_modal = False
        for word, tag in sentence_tree.pos():
            if tag.startswith('V') or tag.startswith('RB'):
                result += '{}-{} '.format(word, tag[0])
            elif tag == 'MD':
                has_modal = True
        if has_modal:
            result += 'MODAL'
        return result
    result = []
    for sentence_tree in matrix:
        result.append(get_filtered_pos(sentence_tree[0]))
    return numpy.array(result)


def get_tree_metrics(matrix):
    """Returns the depth and number of subtrees."""
    result = []
    for sentence_tree in matrix:
        row = [sentence_tree[0].height(),
               len(list(sentence_tree[0].subtrees()))]
        result.append(row)
    return numpy.array(result)


def get_verb_tense(matrix):
    """Returns true if there is a verb in past tense, false otherwise."""
    result = []
    for sentence_tree in matrix:
        row = False
        for _, tag in sentence_tree[0].pos():
            if tag in ['VBN', 'VBD']:
                row = True
                break
        result.append(row)
    return numpy.array(result, ndmin=2).T


def get_structural_features(matrix):
    """Returns a matrix with the structural features of the tree."""
    # Each row is going to be
    # (#tokens, #punct. marks, #pm prev, #pm next, question mark)
    result = []
    previous_pm = 0
    punct_marks = set(string.punctuation)
    for sentence_tree in matrix:
        words = sentence_tree[0].leaves()
        if not words:
            row = [0] * 5
        else:
            punct_marks_count = len(punct_marks.intersection(words))
            row = [
                len(words), punct_marks_count,
                previous_pm, 0, int(words[-1] == '?')
            ]
            if result:  # Add the number of pm for previous sentence
                result[-1][3] = punct_marks_count
        result.append(row)
        previous_pm = row[2]
    return numpy.array(result)


def get_flat_features():
    """Returns the feature extractors for the flatten version of the matrix."""
    return {
        'ngrams': Pipeline([
            ('word_extractor', FunctionTransformer(get_words_from_tree,
                                                   validate=False)),
            ('word_counter', CountVectorizer(
                ngram_range=(1, 3), max_features=10**4)),
            ('tfidf', TfidfTransformer()),
        ]),
        'pos_tags': Pipeline([
            ('pos_extractor', FunctionTransformer(get_pos_from_tree,
                                                  validate=False)),
            ('pos_counter', CountVectorizer(
                ngram_range=(1, 1), max_features=1000))
        ]),
        'tree_metrics': FunctionTransformer(get_tree_metrics, validate=False),
        'verb_tense': FunctionTransformer(get_verb_tense, validate=False),
        'structural_features': FunctionTransformer(get_structural_features,
                                                   validate=False),
    }


def make_flat(x_matrix):
    """Transforms the matrix from a hierarchy of nested lists to an array."""
    new_matrix = []
    for document in x_matrix:
        for paragraph in document:
            for sentence in paragraph:
                new_matrix.append(sentence)
    return new_matrix


def get_tree_steps():
    """Returns the feature extractors for a nltk.tree.Tree input."""
    return {
        'flat': Pipeline([
            ('flatten', FunctionTransformer(make_flat, validate=False)),
            ('extractors', FeatureUnion(get_flat_features().items()))
        ]),
        # 'compositional_features': FunctionTransformer(get_comp_features,
        #                                               validate=False)
    }


def transform_y(y_vector):
    new_labels = []
    for document_labels in y_vector:
        for paragraph_labels in document_labels:
            new_labels.extend(paragraph_labels)
    return new_labels


def get_basic_tree_pipeline(classifier_tuple):
    """Returns an instance of sklearn Pipeline with the preprocess steps.

    The input is expected to be a nltk.tree.Tree instance"""
    features = FeatureUnion(get_tree_steps().items())
    return Pipeline([
        ('features', features),
        classifier_tuple
    ])


def get_tree_parameter_grid():
    """Returns the possible parameters and values for the basic pipeline."""
    features = get_flat_features()

    return {
        # 'features__flat__extractors__transformer_list': [
        #     features.items(),  # All fetures
        #     [('ngrams', features['ngrams']),
        #      ('pos_tags', features['pos_tags'])],
        #     [('ngrams', features['ngrams']),
        #      ('tree_metrics', features['tree_metrics'])],
        #     [('ngrams', features['ngrams']),
        #      ('structural_features', features['structural_features']),
        #      ('verb_tense', features['verb_tense'])],
        #     [('ngrams', features['ngrams']),
        #      ('pos_tags', features['pos_tags']),
        #      ('verb_tense', features['verb_tense'])],
        #     [('ngrams', features['ngrams']),
        #      ('pos_tags', features['pos_tags']),
        #      ('structural_features', features['structural_features'])],
        # ],
        'features__flat__extractors__ngrams__word_counter__max_features': [10**3, 10**4],
        'features__flat__extractors__ngrams__word_counter__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'features__flat__extractors__ngrams__tfidf__use_idf': (True, False)
    }
