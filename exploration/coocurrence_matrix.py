import numpy
from scipy.sparse import lil_matrix
from sklearn import preprocessing


def get_word_index(documents, min_frequency=2):
    """Returns a dictionary mapping from words to unique sequential indices."""
    all_words = []
    for doc in documents:
        all_words.extend([word for sentence in doc.sentences
                          for word in sentence.words])
    counts = {
        word: count
        for word, count in zip(*numpy.unique(all_words, return_counts=True))}
    encoder = preprocessing.LabelEncoder()
    encoder.fit([word for word in all_words if counts[word] >= min_frequency])
    return {word: index for index, word in enumerate(encoder.classes_)}


def word_coocurrence_matrix(documents, indices, window_size=5):
    """Calculates the word coocurrence matrix from a list of documents."""
    matrix = lil_matrix((len(indices), len(indices)))
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
