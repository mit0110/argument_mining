"""Script to preprocess the IBM Debater dataset.

The output is a pickled list of AnnotatedDocuments.

Usage:
    arg_IBM2conll.py --input_dirpath=<dirpath> --output_file=<file> --labels_dirpath=<ldirpath>

Options:
   --input_dirpath=<file>   The path to directory to read files.
   --output_file=<file>         The path to directory to store files.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import fnmatch
import os
import json
import utils
import re
from arg_docs2conll import AnnotatedDocumentFactory
from collections import defaultdict
from preprocess.documents import AnnotatedDocument
from tqdm import tqdm


class AnnotatedIBMFactory(AnnotatedDocumentFactory):
    """Builds a AnnotatedDocument from input_file."""
    DOCUMENT_CLASS = AnnotatedDocument

    def __init__(self, input_filename, identifier=None):
        self.input_filename = input_filename
        self.instance_input_file = None
        self.raw_labels = []
        self.sentences = []
        self.identifier = identifier or input_filename

    def __enter__(self):
        self.instance_input_file = open(self.input_filename, 'r')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.instance_input_file.close()

    def get_labels(self, labels_from_json):
        """Read labels from labels_from_json.
        Saves the resulting labels in self.raw_labels

        Args:
            labels_from_json: A list of tuples with (start, end)
            for this document
        """
        self.raw_labels = labels_from_json

    def build_document(self, labels_from_json):
        """Creates a new AnnotatedDocument instance."""
        if not len(self.raw_labels):
            self.get_labels(labels_from_json)
        raw_text = self.instance_input_file.read()
        document = self.DOCUMENT_CLASS(self.identifier, title=self.identifier)
        document.build_from_text(raw_text, start_index=0)
        for start_index, end_index in self.raw_labels:
            document.add_label_for_position(
                'claim', int(start_index), int(end_index))
        return document

def get_all_labels_from_json(labels_dirname):
    """Returns a tuple with the list of original claims and their start and end.
    """
    labels_from_json={}
    labels_from_json = defaultdict(list)
    with open(labels_dirname) as f:
        data = json.load(f)
        for topicTarget in data:
            for claim in topicTarget['claims']:
                start = claim['article']['cleanSpan']['start']
                end = claim['article']['cleanSpan']['end']
                claim_start_end = (start, end)
                key = claim['article']['cleanFile']
                labels_from_json[key].append(claim_start_end)
    return labels_from_json

def traverse_directory(path, file_pattern='*'):
    for root, _, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, file_pattern):
            yield os.path.join(root, filename)

def main():
    """Main fuction of the script."""
    args = utils.read_arguments(__doc__)
    documents = []
    filenames = list(traverse_directory(args["input_dirpath"],'*clean*.txt'))
    labels_dirname = args["labels_dirpath"]
    labels_from_json = get_all_labels_from_json(labels_dirname)
    for filename in tqdm(filenames):
        with AnnotatedIBMFactory(filename) as instance_extractor:
            filename_key = "/".join(filename.split("/")[-3:])
            document = instance_extractor.build_document(
                labels_from_json[filename_key])

            documents.append(document)

    utils.pickle_to_file(documents, args['output_file'])

if __name__ == '__main__':
    main()
