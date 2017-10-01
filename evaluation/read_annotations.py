"""Functions to read annotations and convert them in AnnotatedDocument instances"""

import itertools
import os
import re
import sys

from collections import defaultdict

def append_path(module_path):
    if module_path not in sys.path:
        sys.path.append(module_path)

append_path(os.path.abspath('..'))

from preprocess import annotated_documents, arg_docs2conll


ANNOTATION_FORMAT = r'.*\.ann'


# Find files to compare
def get_non_empty_filenames(input_dirpath, pattern, size_limit=500):
    """Returns the names of the files in input_dirpath matching pattern."""
    all_files = os.listdir(input_dirpath)
    result = {}
    for filename in all_files:
        if not re.match(pattern, filename):
            continue
        filepath = os.path.join(input_dirpath, filename)
        if os.path.isfile(filepath) and os.stat(filepath).st_size > 500:
            result[filename] = filepath
    return result


def get_filenames_by_document(annotations_dir, annotators):
    files = defaultdict(lambda: {})
    for name, annotator in annotators.items():
        annotator['files'] = get_non_empty_filenames(
            os.path.join(annotations_dir, annotator['dirname']),
            ANNOTATION_FORMAT)
        for filename, filepath in annotator['files'].items():
            files[filename][name] = filepath
    return dict(files)


def get_filenames_by_annotator(annotations_dir, annotators):
    filenames = defaultdict(list)
    for name, annotator in annotators.items():
        for filename in get_non_empty_filenames(
                os.path.join(annotations_dir, annotator['dirname']),
                ANNOTATION_FORMAT).values():
            filenames[name].append(filename)
    return dict(filenames)


def get_annotation(filename, annotator_name):
    identifier = 'Case: {} - Ann: {}'.format(
        os.path.basename(filename[:-4]).replace(
            'CASE_OF__', '').replace('_', ' '),
        annotator_name[0].title())
    txt_filename = filename.replace('.ann', '.txt')
    with arg_docs2conll.AnnotatedJudgementFactory(
            txt_filename, identifier) as instance_extractor:
        try:
            document = instance_extractor.build_document()
        except Exception as exc:
            print('Error processing document {}'.format(filename))
            raise exc
    return document


def get_all_documents(annotations_dir, annotators):
    filenames = get_filenames_by_annotator(annotations_dir, annotators)
    documents = defaultdict(list)
    for annotator, annotator_filenames in filenames.items():
        for filename in annotator_filenames:
            documents[annotator].append(get_annotation(filename, annotator))
    return documents


def get_annotated_documents(annotations_dir, annotators):
    files = get_filenames_by_document(annotations_dir, annotators)
    document_pairs = []
    for value in files.values():
        if len(value) < 2:
            continue
        annotations = read_parallel_annotations(value.items())
        for ann1, ann2 in list(itertools.combinations(annotations.keys(), 2)):
            document_pairs.append((annotations[ann1], annotations[ann2]))
    return document_pairs, annotations


def read_parallel_annotations(annotator_filenames):
    annotations = {}
    for name, filename in annotator_filenames:
        annotations[name] = get_annotation(filename, name) 
    return annotations


def get_labels(doc1, doc2):
    words1, labels1 = doc1.get_word_label_list()
    words2, labels2 = doc2.get_word_label_list()
    # Check the documents are equal
    assert words1 == words2
    return labels1, labels2
