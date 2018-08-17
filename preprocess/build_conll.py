"""Script to transform AnnotatedDocument to Conll format including relations.

Output format: CONLL labels (tab separated)

index   token    [BI]-[entity label]:[related entity's index]:[relation label]


Usage:
    build_conll.py --input_filename=<filename> --output_dirname=<filename> [--include_relations] [--separation=<separation>]

Options:
    --input_filename <filename>     The path to pickled file contianing the
                                    documents.
    --output_dirname <dirname>      The path to the directory to store the
                                    resulting conll files
    --include_relations             Weather to include relations in the label or
                                    not.
    --separation <separation>       The separation level. Options are sentence,
                                    paragraph or section. Default is sentence.

"""
from __future__ import print_function

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import utils


class DocumentWriter(object):
    def __init__(self, output_dirname, include_relations=True, separation=None):
        self.output_dirname = output_dirname
        self.token_index = 0
        self.include_relations = include_relations
        self.relations = None
        self.component_starts = {}
        self.last_component_start = 0
        self.document = None
        self.separation = separation
        self.current_paragraph = None
        self.current_section = None

    @staticmethod
    def is_begin(word_start, word, document):
        """Returns True if the word is the start of a component."""
        for possible_start in range(word_start, word_start+len(word)):
            if possible_start in document.annotated_components:
                return True
        return False

    def write_document(self, document):
        self.token_index = 0
        # Map from component start in document level to line index in conll
        self.relations = document.get_relative_relations()
        self.component_starts = {}
        self.last_component_start = 0
        self.document = document
        self.current_paragraph = None
        self.current_section = None

        for sentence in document.sentences:
            if (self.separation == 'paragraph'
                  and sentence.paragraph_number != self.current_paragraph):
                self.current_paragraph = sentence.paragraph_number
                self.end_section()
            elif (self.separation == 'section'
                  and sentence.section != self.current_section):
                self.current_section = sentence.section
                self.end_section()
            self._write_sentence(sentence)
            if self.separation == 'sentence':
                self.end_section()
        self.end_section()

    def _write_sentence(self, sentence):
        for word_index, word in enumerate(sentence.words):
            relation = None
            target_index = None
            label = sentence.labels[word_index]
            bio_label = 'O'
            if label != self.document.default_label:
                bio_label = 'I'
                word_start = sentence.word_positions[word_index]
                if self.is_begin(word_start, word, self.document):
                    # Start of a new component
                    bio_label = 'B'
                    self.last_component_start = word_start
                    self.component_starts[word_start] = self.token_index
                if self.include_relations:
                    relation, target_index = self._get_relations(
                        self.last_component_start, self.relations)
            self._write_line(word, bio_label, label, relation, target_index,
                             output_file)

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

    def _write_line(self, word, bio_label, label, relation, target_index,
                    output_file):
        if bio_label == 'O':
            output_file.write('{}\t{}\t_\t_\t{}\n'.format(
                self.token_index, word, bio_label))
        elif not self.include_relations:
            output_file.write('{}\t{}\t_\t_\t{}-{}\n'.format(
                self.token_index, word, bio_label, label))
        else:
            output_file.write('{}\t{}\t_\t_\t{}-{}:{}:{}\n'.format(
                self.token_index, word, bio_label, label,
                relation, target_index))
        self.token_index += 1

    def end_section(self, output_file):
        output_file.write('\n')


def main():
    """Main function of script"""
    args = utils.read_arguments(__doc__)
    documents = utils.pickle_from_file(args['input_filename'])
    if args['separation'] in ['sentence', 'paragraph', 'section']:
        separation = args['separation']
    else:
        separation = 'sentence'

    writer = DocumentWriter(output_dirname=args['output_dirname'],
                            include_relations=args['include_relations'],
                            separation=separation)
    for document in documents:
        if document.has_annotation():
            print('Adding document {}'.format(document.identifier))
            writer.write_document(document)



if __name__ == '__main__':
    main()
