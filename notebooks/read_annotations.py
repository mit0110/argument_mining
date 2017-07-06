"""Functions to read annotations and convert them in EssayDocument instances"""

import itertools
import os
import re
import sys

from collections import defaultdict

def append_path(module_path):
    if module_path not in sys.path:
        sys.path.append(module_path)

append_path(os.path.abspath('..'))

from preprocess import essay_documents, process_arg_essays_for_conll

ANNOTATIONS_DIR = '/home/milagro/am/third_party/brat-v1.3_Crunchy_Frog/data/'
ANNOTATORS = {
    'mili': {'dirname': 'judgements-mili'},
    # 'laura': {'dirname': 'judgements-laura'},
    'serena': {'dirname': 'judgements-serena'}
}
ANNOTATION_FORMAT = r'.*\.ann'
BRAT_DIRNAME = '/home/milagro/FaMAF/am/third_party/brat/'


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

def get_annotated_filenames():
    files = defaultdict(lambda: {})
    for name, annotator in ANNOTATORS.items():
        annotator['files'] = get_non_empty_filenames(
            os.path.join(ANNOTATIONS_DIR, annotator['dirname']),
            ANNOTATION_FORMAT)
        for filename, filepath in annotator['files'].items():
            files[filename][name] = filepath
    return files

def get_annotated_documents():
    files = get_annotated_filenames()
    document_pairs = []
    for value in files.values():
        if len(value) < 2:
            continue
        annotations = {}
        for name, filename in value.items():
            identifier = 'Case: {} - Ann: {}'.format(
                os.path.basename(filename[:-4]).replace(
                    'CASE_OF__', '').replace('_', ' '),
                name[0].title())
            with process_arg_essays_for_conll.EssayDocumentFactory(
                    filename.replace('ann', 'txt'), identifier) as instance_extractor:
                annotations[name] = instance_extractor.build_document()
        for ann1, ann2 in list(itertools.combinations(annotations.keys(), 2)):
            document_pairs.append((annotations[ann1], annotations[ann2]))
    return document_pairs


def get_labels(doc1, doc2):
    words1, labels1 = doc1.get_word_label_list()
    words2, labels2 = doc2.get_word_label_list()
    # Check the documents are equal
    assert words1 == words2
    return labels1, labels2
