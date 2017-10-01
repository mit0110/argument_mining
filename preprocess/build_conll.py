"""Script to transform AnnotatedDocument to Conll format including relations.

Output format: CONLL labels (tab separated)

index   token    [BI]-[entity label]:[related entity's index]:[relation label]


Usage:
    build_conll.py --input_filename=<filename> --output_filename=<filename> [--include_relations]

Options:
    --input_filename <filename>     The path to pickled file contianing the
                                    documents.
    --output_filename <filename>    The path to the conll file to store the
                                    output
    --include_relations             Weather to include relations in the label or not.

"""
from __future__ import print_function

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import utils


class DocumentWriter(object):
    def __init__(self, output_file, include_relations=True):
        self.output_file = output_file
        self.token_index = 0
        self.include_relations = include_relations
        self.relations = None
        self.component_starts = {}
        self.last_component_start = 0
        self.document = None

    @staticmethod
    def is_begin(word_start, document, word):
        """Returns True if the word is the start of a component."""
        return word_start in document.annotated_components

    def write_document(self, document):
        self.token_index = 0
        # Map from component start in document level to line index in conll
        self.relations = document.get_relative_relations()
        self.document = document
        for sentence in document.sentences:
            self._write_sentence(sentence)

    def _write_sentence(self, sentence):
        for word_index, word in enumerate(sentence.words):
            relation = None
            target_index = None
            label = sentence.labels[word_index]
            bio_label = 'O'
            if label != self.document.default_label:
                bio_label = 'I'
                word_start = sentence.word_positions[word_index]
                if self.is_begin(word_start, self.document, word):
                    # Start of a new component
                    bio_label = 'B'
                    self.last_component_start = word_start
                    self.component_starts[word_start] = self.token_index
                if self.include_relations:
                    relation, target_index = self._get_relations(
                        self.last_component_start, self.relations)
            self._write_line(word, bio_label, label, relation, target_index)

    @staticmethod
    def _get_relations(component_start, relations):
        target_dict = relations.get(component_start, None)
        if target_dict is None:
            return None, None
        if len(target_dict) == 1:
            return list(target_dict.items())[0]
        for target, relation in target_dict.items():
            if relation in ['Attack', 'Support']:
                return target, relation

    def _write_line(self, word, bio_label, label, relation, target_index):
        if bio_label == 'O':
            self.output_file.write('{}\t{}\t{}\n'.format(
                self.token_index, word, bio_label))
        elif not self.include_relations:
            self.output_file.write('{}\t{}\t{}-{}\n'.format(
                self.token_index, word, bio_label, label))
        else:
            self.output_file.write('{}\t{}\t{}-{}:{}:{}\n'.format(
                self.token_index, word, bio_label, label,
                relation, target_index))
        self.token_index += 1


def main():
    """Main function of script"""
    args = utils.read_arguments(__doc__)
    documents = utils.pickle_from_file(args['input_filename'])
    with open(args['output_filename'], 'w') as output_file:
        writer = DocumentWriter(output_file,
                                include_relations=args['include_relations'])
        for document in documents:
            if document.has_annotation():
                writer.write_document(document)


if __name__ == '__main__':
    main()
