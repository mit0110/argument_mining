"""Script to preprocess the Argumentative Essays dataset.

The output is a pickled list of AnnotatedDocuments.

Usage:
    arg_docs2conll.py --input_dirpath=<dirpath> --output_file=<file> [--limit=<N>] [--parse_trees]

Options:
   --input_dirpath=<dirpath>    The path to directory to read files.
   --output_file=<file>         The path to directory to store files.
   --limit=<N>                  The number of files to read. -1 for all. [default: -1]
   --parse_trees                Apply Stanford parser to documents
"""

import logging
logging.basicConfig(level=logging.INFO)
import re
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import utils
from tqdm import tqdm
from preprocess.annotated_documents import AnnotatedDocument, AnnotatedJudgement
from preprocess.lexicalized_stanford_parser import LexicalizedStanfordParser


class AnnotatedDocumentFactory(object):
    """Builds a AnnotatedDocument from input_file."""

    DOCUMENT_CLASS = AnnotatedDocument

    def __init__(self, input_filename, identifier=None):
        self.input_filename = input_filename
        self.label_input_file = None
        self.instance_input_file = None
        self.raw_labels = {}
        self.sentences = []
        self.identifier = identifier or input_filename
        self.title = ''
        self.raw_relations = []

    def __enter__(self):
        self.label_input_file = open(
            self.input_filename.replace('txt', 'ann'), 'r')
        self.instance_input_file = open(self.input_filename, 'r')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.label_input_file.close()
        self.instance_input_file.close()

    def get_labels(self):
        """Read labels from the annotation file.

        Saves the resulting labels in self.raw_labels
        """
        self.raw_labels = {}
        for line in self.label_input_file.readlines():
            if line == '':
                continue
            if line.startswith('T'):
                self.process_component(line)
            elif line.startswith('R'):
                self.process_relation(line)

    def process_relation(self, line):
        label_info = line.split('\t', 2)[1]
        label, arg1, arg2 = label_info.split(' ')
        arg1 = arg1.replace('Arg1:', '')
        arg2 = arg2.replace('Arg2:', '')
        self.raw_relations.append((arg1, arg2, label))

    def process_component(self, line):
        component_name, label, _ = line.split('\t', 2)
        # label has shape label start end;start end; ...
        if ';' in label:  # The label has several fragments:
            label_name, indices = label.split(' ', 1)
            label_fragments = []
            for fragment in indices.split(';'):
                start, end = fragment.split()
                label_fragments.append((label_name, start, end))
        else:
            label_fragments = [label.split()]
        self.raw_labels[component_name] = label_fragments

    def build_document(self):
        """Creates a new AnnotatedDocument instance."""
        if not len(self.raw_labels):
            self.get_labels()
        title = self.instance_input_file.readline()
        content = title + self.instance_input_file.read()
        document = self.DOCUMENT_CLASS(self.identifier, title=title)
        document.build_from_text(content, start_index=0)
        # Add components
        for component, fragments in self.raw_labels.items():
            first_start = fragments[0][1]
            for label, start_index, end_index in fragments:
                document.add_label_for_position(label, int(start_index),
                                                int(end_index))
            document.add_component(component, int(first_start), int(end_index))
        # Add relations
        for arg1, arg2, label in self.raw_relations:
            document.add_relation(label, arg1, arg2)
        return document


class AnnotatedJudgementFactory(AnnotatedDocumentFactory):

    DOCUMENT_CLASS = AnnotatedJudgement
    def __init__(self, input_filename, identifier=None):
        super(AnnotatedJudgementFactory, self).__init__(
            input_filename, identifier)
        self.raw_attributes = {}

    def get_labels(self):
        """Read labels from the annotation file.
        Saves the resulting labels in self.raw_labels
        """
        self.raw_labels = {}
        for line in self.label_input_file.readlines():
            if line == '':
                continue
            if line.startswith('T'):
                self.process_component(line)
            elif line.startswith('R'):
                self.process_relation(line)
            elif line.startswith('A'):
                self.process_attribute(line)

    def process_attribute(self, line):
        attribute_info = line.split('\t', 2)[1]
        name, component_name, value = attribute_info.split()
        # label has shape label start end;start end; ...
        self.raw_attributes[component_name] = (component_name, value)

    def build_document(self):
        """Creates a new AnnotatedDocument instance."""
        if not len(self.raw_labels):
            self.get_labels()
        title = self.instance_input_file.readline()
        content = title + self.instance_input_file.read()
        document = self.DOCUMENT_CLASS(self.identifier, title=title)
        document.build_from_text(content, start_index=0)
        # Add components
        for component, fragments in self.raw_labels.items():
            attribute = self.raw_attributes.get(component, None)
            first_start = fragments[0][1]
            for label, start_index, end_index in fragments:
                document.add_label_for_position(
                    label, int(start_index), int(end_index), attribute)
            document.add_component(component, int(first_start), int(end_index))
        # Add relations
        for arg1, arg2, label in self.raw_relations:
            document.add_relation(label, arg1, arg2)
        return document


def get_input_files(input_dirpath, pattern, limit=-1):
    """Returns the names of the files in input_dirpath that matches pattern."""
    all_files = os.listdir(input_dirpath)
    if limit < 0:
        limit = len(all_files)
    result = []
    for filename in all_files:
        if re.match(pattern, filename) and os.path.isfile(os.path.join(
                input_dirpath, filename)):
            result.append(os.path.join(input_dirpath, filename))
        if len(result) >= limit:
            break
    return result


def main():
    """Main fuction of the script."""
    args = utils.read_arguments(__doc__)
    documents = []
    filenames = get_input_files(args['input_dirpath'], r'.*txt',
                                int(args['limit']))
    if args['parse_trees']:
        parser = LexicalizedStanfordParser(
            model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',
            encoding='utf8')
    else:
        parse = None
    for filename in tqdm(filenames):
        with AnnotatedJudgementFactory(filename) as instance_extractor:
            document = instance_extractor.build_document()
            if args['parse_trees']:
                document.parse_text(parser)
            documents.append(document)

    utils.pickle_to_file(documents, args['output_file'])


if __name__ == '__main__':
    main()
