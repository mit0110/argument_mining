"""Script to transform AnnotatedDocument to Conll format including relations.

Output format: CONLL labels (tab separated)

index   token    [BI]-[entity label]:[related entity's index]:[relation label]


Usage:
    build_conll.py --input_filename=<filename> --output_filename=<filename>

Options:
    --input_filename <filename>     The path to pickled file contianing the
                                    documents.
    --output_filename <filename>    The path to the conll file to store the
                                    output

"""
from __future__ import print_function

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../preprocess/'))

import utils


class DocumentWriter(object):
    def __init__(self, output_file):
        self.output_file = output_file
        self.token_index = 0

    @staticmethod
    def is_begin(word_start, document, word):
        """Returns True if the word is the start of a component."""
        return word_start in document.annotated_components

    def write_document(self, document):
        self.token_index = 0
        # Map from component start in document level to line index in conll
        component_starts = {}
        relations = document.get_relative_relations()
        last_component_start = 0
        for sentence in document.sentences:
            for word_index, word in enumerate(sentence.words):
                relation = None
                target_index = None
                label = sentence.labels[word_index]
                bio_label = 'O'
                if label != document.default_label:
                    bio_label = 'I'
                    word_start = sentence.word_positions[word_index]
                    if self.is_begin(word_start, document, word):
                        # Start of a new component
                        bio_label = 'B'
                        last_component_start = word_start
                        component_starts[word_start] = self.token_index
                    relation, target_index = self._get_relations(
                        last_component_start, relations)
                self._write_line(word, bio_label, label, relation, target_index)

    def _get_relations(self, component_start, relations):
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
        writer = DocumentWriter(output_file)
        for document in documents:
            writer.write_document(document)


if __name__ == '__main__':
    main()