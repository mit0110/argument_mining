"""Base preprocess pipeline common for all classifiers."""

import nltk
import numpy

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
    return result#.reshape(1, -1)


def get_pos_steps():
    return [
        # Word features
        ('vect', CountVectorizer(ngram_range=(1, 3), max_features=10**4)),
        # Pos tag features
        ('pos', Pipeline([
            ('pos_extractor', FunctionTransformer(pos_filter)),
            ('pos_vectorizers', FeatureUnion([
                # ('pos_vect', CountVectorizer(
                #     ngram_range=(1, 1), max_features=1000,
                #     token_pattern=u'\[\[.*?\]\]')),
                ('modal_extractor', FunctionTransformer(get_model_verbs)),
            ]))
        ]))
    ]


def get_basic_pipeline(classifier_tuple):
    """Returns an instance of sklearn Pipeline with the preprocess steps."""
    features = FeatureUnion(get_pos_steps())
    return Pipeline([
            ('pos_extractor', FunctionTransformer(pos_filter)),
            ('pos_vectorizers', FeatureUnion([
                ('pos_vect', CountVectorizer(
                    ngram_range=(1, 1), max_features=1000,
                    token_pattern=u'\[\[.*?\]\]')),
                ('modal_extractor', FunctionTransformer(get_model_verbs)),
            ])),
            classifier_tuple
        ])
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
