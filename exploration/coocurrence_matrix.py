import numpy

from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from sklearn import preprocessing

from nltk.corpus import wordnet


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 


def get_word_index(documents, min_frequency=2):
    """Returns a dictionary mapping from words to unique sequential indices."""
    lemmatizer = WordNetLemmatizer()
    counts = defaultdict(lambda: 0)
    for doc in documents:
        for sentence in doc.sentences:
            for word, pos in zip(sentence.words, sentence.pos):
                counts[(word, pos)] += 1
    lemma_ids = {}
    word_ids = {}
    ids_to_word = {}
    for (word, pos), count in counts.items():
        if count < min_frequency:
            continue
        pos = get_wordnet_pos(pos)
        lemma = '{}-{}'.format(lemmatizer.lemmatize(word, pos=pos), pos)
        if lemma not in lemma_ids:
            lemma_ids[lemma] = len(lemma_ids)

        word_ids[word] = lemma_ids[lemma]
        ids_to_word[lemma_ids[lemma]] = lemma

    return word_ids, ids_to_word


def word_coocurrence_matrix(documents, indices, window_size=5):
    """Calculates the word coocurrence matrix from a list of documents."""
    max_index = numpy.max(list(indices.values())) + 1
    matrix = lil_matrix((max_index, max_index))
    for doc in documents:
        words = [word for sentence in doc.sentences for word in sentence.words]
        for word_position, word in enumerate(words):
            if word not in indices:
                continue
            word_id = indices[word]
            for next_word in words[word_position + 1:
                                   word_position + window_size]:
                if next_word not in indices:
                    continue
                next_word_id = indices[next_word]
                matrix[word_id, next_word_id] += 1
                matrix[next_word_id, word_id] += 1
    return matrix


class wordKDTree(object):
    def __init__(self, matrix):
        self.tree = cKDTree(matrix)
    
    def get_closest(self, target_vectors, k, distance_norm=2):
        _, indices = self.tree.query(target_vectors, k, p=distance_norm)
        if k > 1:
            indices = indices.squeeze()
        return indices

