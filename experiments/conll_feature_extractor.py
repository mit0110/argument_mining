"""Functions to convert conll formated EssayDocument into a numeric matrix."""

import itertools
import numpy
import string

from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm

PUNCTUATION_MARKS = set(string.punctuation)


def split_lexical_node(label):
    """Splits a lexical node label into component label, token and pos tag.

    For example, 'PP[From/IN]' -> 'PP', 'From', 'IN'
    """
    tag, token_pos = label.split('[')
    token, pos = token_pos.strip(']').split('/')
    return tag, token, pos


def get_neightbor_lca(leaf_position, tree, following=True):
    """Gets the lenght and label of the path to the Lowest Common Ancestor.

    Args:
        leaf_position, integer. Position of the word in the
            tree leaves.
        tree: a nltk.tree.Tree instance.
        following: if True, the distance is computed with the following
            neightbor

    If the position is in the edge of the tree, the distance is the height of
    the tree plus one and the label is an empty string.
    Based in:
    http://stackoverflow.com/questions/28681741/find-a-path-in-an-nltk-tree-tree
    """
    if following and leaf_position >= len(tree.leaves()) - 1:
        return tree.height() + 1, ''
    if (not following) and leaf_position < 1:
        return tree.height() + 1, ''
    index = 0
    location1 = tree.leaf_treeposition(leaf_position)
    location2 = tree.leaf_treeposition(leaf_position + (1 if following else -1))
    while (index < len(location1) and index < len(location2)
            and location1[index] == location2[index]):
        index += 1
    tag = tree[location1[:index]].label()
    return index, split_lexical_node(tag)[0]


def get_ancestor_by_tag(tree, start_position):
    """Returns the upper most ancestor node of start_position where the label
    of start_position is contained in the label of the ancestor."""
    position = tree.leaf_treeposition(start_position)
    label = tree[position]
    if hasattr(label, 'label'):  # start position is not a leaf
        label = label.label()
    ancestor_position = position
    for index in range(1, len(position)):
        if label not in tree[position[:-1*index]].label():
            if index == 1:  # The parent has a different label
                return label
            return tree[ancestor_position].label()
        ancestor_position = ancestor_position[:-1]

def get_parent_sibling(parse_tree, start_position):
    """Returns the labels of the parent and the right sibling of start_position.
    """
    parent_position = parse_tree.leaf_treeposition(start_position)[:-1]
    right_position = parent_position[:-1] + (parent_position[-1] + 1,)
    if (start_position < len(parse_tree.leaves()) - 1 and
            right_position in parse_tree.treepositions()):
        return (parse_tree[parent_position].label(),
                parse_tree[right_position].label())
    return None


class ConllFeatureExtractor(object):
    """Converts EssayDocument list into numeric matrix."""

    def __init__(self, use_structural=True, use_syntactic=False,
                 use_lexical=False):
        self.use_structural = use_structural
        self.use_syntactic = use_syntactic
        self.use_lexical = use_lexical

    def get_structural_features(self, document):
        """Adds structural features to the dictionary features."""
        instances = []
        if not self.use_structural:
            return instances
        word_position_in_paragraph = 0
        word_position_in_document = 0
        for sent_index, sentence in enumerate(document.sentences):
            sentence_instances = []
            if sentence.position_in_paragraph == 0:
                word_position_in_paragraph = 0
            for token_index, token, _ in sentence.iter_words():
                features = defaultdict(int)
                # Is the word in the introduction or the conclusion?
                features['st:tk:introduction'] = sentence.paragraph_number == 0
                features['st:tk:conclusion'] = (sentence.paragraph_number ==
                    document.sentences[-1].paragraph_number)
                # Position of word in sentence
                features['st:tk:position_in_sentence'] = token_index
                features['st:tk:first'] = token_index == 0
                features['st:tk:last'] = token_index == len(sentence) - 1
                # Position in paragraph
                features['st:tk:pos_in_paragraph'] = word_position_in_paragraph
                word_position_in_paragraph += 1
                # Position in document
                features['st:tk:pos_in_document'] = word_position_in_document
                word_position_in_document += 1

                # Punctuation marks
                features['st:pn:is_full_stop'] = (token == '.' and
                    token_index == len(sentence) - 1)
                features['st:pn:is_puntuation'] = token in PUNCTUATION_MARKS
                if token_index > 0:
                    prev_token = sentence.words[token_index - 1]
                elif sent_index > 0:
                    prev_token = document.sentences[sent_index - 1].words[-1]
                    features['st:pn:follows_full_stop'] = prev_token == '.'
                else:
                    prev_token = ''
                features['st:pn:follows'] = prev_token in PUNCTUATION_MARKS
                features['st:pn:follows_comma'] = prev_token == ','
                features['st:pn:follows_semicolon'] = prev_token == ';'
                # Is not the first word in the document
                if not (sent_index == 0 and token_index == 0):
                    # Is the first word on the sentence
                    if len(sentence_instances) == 0:
                        last_feature = instances[-1][-1]
                    else:
                        last_feature = sentence_instances[-1]
                    last_feature['st:pn:preceeds'] = token in PUNCTUATION_MARKS
                    last_feature['st:pn:preceeds_comma'] = token == ','
                    last_feature['st:pn:preceeds_semicolon'] = token == ';'
                    if features['st:tk:last']:
                        last_feature['st:pn:preceeds_full_stop'] = token == '.'

                # Position of covering sentence
                features['st:cs:position_in_paragraph'] = (
                    sentence.position_in_paragraph)
                features['st:cs:position_in_document'] = (
                    sentence.position_in_document)

                sentence_instances.append(features)
            instances.append(sentence_instances)
        return instances

    def get_syntactic_features(self, document):
        """Returns a list of dictionaries with syntactic features."""
        if not self.use_syntactic:
            return []
        instances = []
        for sent_index, sentence in enumerate(document.sentences):
            parse_tree = document.parse_trees[sent_index]
            sentence_instances = []
            for token_index, _, pos_tag in sentence.iter_words():
                features = defaultdict(int)
                features['syn:pos'] = pos_tag
                lca_distance, lca_tag = get_neightbor_lca(
                    token_index, parse_tree, following=True)
                features['syn:lca:next'] = lca_distance
                features['syn:lca:next_tag'] = lca_tag
                lca_distance, lca_tag = get_neightbor_lca(
                    token_index, parse_tree, following=False)
                features['syn:lca:prev'] = lca_distance
                features['syn:lca:prev_tag'] = lca_tag
                sentence_instances.append(features)
            instances.append(sentence_instances)
        return instances

    def get_lexical_features(self, document):
        """Returns a list of dictionaries with syntactic features."""
        if not self.use_lexical:
            return []
        instances = []
        for sent_index, sentence in enumerate(document.sentences):
            parse_tree = document.parse_trees[sent_index]
            sentence_instances = []
            for token_index, _, _ in sentence.iter_words():
                features = defaultdict(int)
                # Lexical head
                features['ls:token_comb'] = get_ancestor_by_tag(parse_tree,
                                                                token_index)
                # parent's sibling
                labels = get_parent_sibling(parse_tree, token_index)
                if labels:
                    features['ls:right_comb'] = '-'.join(labels)
                sentence_instances.append(features)
            instances.append(sentence_instances)
        return instances

    @staticmethod
    def combine_features(features_list):
        """Combines a list of list of feature dictionaries row by row."""
        sentence_number = max([len(features) for features in features_list])
        # Create a list for each sentence. Inside the list, it will be an
        instances = [[] for _ in range(sentence_number)]
        for feature_type in features_list:
            assert (len(feature_type) == sentence_number
                    or len(feature_type) == 0)
            if len(feature_type) == 0:
                continue
            for sentence_index, sentence_features in enumerate(feature_type):
                if len(instances[sentence_index]) == 0:
                    instances[sentence_index] = [
                        {} for _ in range(len(sentence_features))]
                for word_index, feature_dictionary in enumerate(
                        sentence_features):
                    instances[sentence_index][word_index].update(
                        feature_dictionary)
        return instances

    def get_feature_dict(self, documents):
        """Returns a list of dictionaries of features, one per instance."""
        instances = []
        for document in tqdm(documents):
            structural_features = self.get_structural_features(document)
            syntactic_features = self.get_syntactic_features(document)
            lexical_features = self.get_lexical_features(document)
            instances.extend(self.combine_features(
                [structural_features, syntactic_features, lexical_features]))
        return instances

    def transform(self, documents):
        """Returns a numpy array with extracted features."""
        instances = list(itertools.chain(*self.get_feature_dict(documents)))
        # Convert to matrix and return

        vectorizer = DictVectorizer(dtype=numpy.int32)
        dataset_matrix = vectorizer.fit_transform(instances)

        return dataset_matrix


def get_labels_from_documents(documents):
    """Returns a concatenation of labels of the sentences in documents."""
    labels = []
    for document in documents:
        for sentence in document.sentences:
            labels.append(sentence.labels)
    return labels
