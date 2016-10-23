"""Base preprocess pipeline common for all classifiers."""

import itertools
import nltk
import numpy

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
    for sentence_tree in matrix:
        result.append(' '.join(sentence_tree[0].leaves()))
    return numpy.array(result)


def get_pos_from_tree(matrix):
    """Returns a string with PoS tags from the tree"""
    def get_filtered_pos(sentence_tree):
        """Returns verb, adverb (pos tags and word) and modal (pos tags)"""
        result = ''
        has_modal = False
        for word, tag in sentence_tree.pos():
            if tag.startswith('V') or tag == 'RB':
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


def get_tree_steps():
    """Returns the feature extractors for a nltk.tree.Tree input."""
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
        'tree_metrics': FunctionTransformer(get_tree_metrics, validate=False)
    }


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
    features = get_tree_steps()

    return {
        'features__transformer_list': [
            features.items(),  # All fetures
            [('ngrams', features['ngrams']),
             ('pos_tags', features['pos_tags'])],
            [('ngrams', features['ngrams']),
             ('tree_metrics', features['tree_metrics'])],
        ],
        'features__ngrams__word_counter__max_features': [10**3, 10**4, 10**5],
        'features__ngrams__word_counter__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'features__ngrams__word_counter__tfidf__use_idf': (True, False)
    }
