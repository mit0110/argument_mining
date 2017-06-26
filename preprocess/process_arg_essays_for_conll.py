"""Script to preprocess the Argumentative Essays dataset.

The output is a pickled list of EssayDocuments.

Usage:
    process_arg_essays_for_conll.py --input_dirpath=<dirpath> --output_file=<file> [--limit=<N>] [--parse_trees]

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
from essay_documents import EssayDocument
from lexicalized_stanford_parser import LexicalizedStanfordParser


class EssayDocumentFactory(object):
    """Builds a EssayDocument from input_file."""
    def __init__(self, input_filename):
        self.input_filename = input_filename
        self.label_input_file = None
        self.instance_input_file = None
        self.raw_labels = []
        self.sentences = []
        self.title = ''

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

        Each label is a map
            start_index -> (label type, component text).
        Saves the resulting labels in self.raw_labels
        """
        self.raw_labels = []
        for line in self.label_input_file.readlines():
            if line == '' or not line.startswith('T'):
                continue
            label_info = line.split('\t')[1]
            self.raw_labels.append(label_info.split())

    def build_document(self):
        """Creates a new EssayDocument instance."""
        if not len(self.raw_labels):
            self.get_labels()
        title = self.instance_input_file.readline()
        content = self.instance_input_file.read()
        document = EssayDocument(os.path.basename(self.input_filename),
                                 title=title)
        document.build_from_text(content, start_index=len(title) + 1)
        for label, start_index, end_index in self.raw_labels:
            document.add_label_for_position(label, int(start_index),
                                            int(end_index))
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
    parser = LexicalizedStanfordParser(
        model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',
        encoding='utf8')
    for filename in tqdm(filenames):
        with EssayDocumentFactory(filename) as instance_extractor:
            document = instance_extractor.build_document()
            if args['parse_trees']:
                document.parse_text(parser)
            documents.append(document)

    utils.pickle_to_file(documents, args['output_file'])


if __name__ == '__main__':
    main()
