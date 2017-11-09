"""Convert the ECHR unlabeled documents into UnlabeledDocument instances.

The output is a pickled list of AnnotatedDocuments.

Usage:
    build_unlabeled_docs.py --input_file=<ifile> --output_file=<ofile> [--limit=<N>] [--parse_trees]

Options:
   --input_file=<ifile>     The path to directory to read files.
   --output_file=<ofile>    The path to directory to store files.
   --limit=<N>              The number of files to read. -1 to read
                            all files. [default: -1]
   --parse_trees            Apply Stanford parser to documents
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import utils

from preprocess.documents import UnlabeledDocument
from preprocess.lexicalized_stanford_parser import LexicalizedStanfordParser


def _is_new_document(line):
    return line.startswith('CASE OF  ')


def _create_document(text_buffer, parser=None):
    new_document = UnlabeledDocument(text_buffer[0])
    new_document.build_from_text(text_buffer)
    if parser is not None:
        new_document.parse_text(parser)
    return new_document


def main():
    """Main fuction of the script."""
    args = utils.read_arguments(__doc__)
    documents = []
    text_buffer = []
    if args['parse_trees']:
        parser = LexicalizedStanfordParser(
            model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',
            encoding='utf8')
    else:
        parser = None
    limit = int(args['limit'])

    with open(args['input_file'], 'r') as input_file:
        for line in input_file:
            if _is_new_document(line) and len(text_buffer) > 0:
                try:
                    documents.append(_create_document(text_buffer, parser))
                except Exception as e:
                    print('Creation failed for document: {}'.format(
                        text_buffer[0]))
                    print(e)
                text_buffer = [line]
                print('Adding case {} {}'.format(len(documents), line.strip()))
                if limit > 0 and len(documents) >= limit:
                    break

                if len(documents) % 10 == 0:  # Partial save
                    utils.pickle_to_file(documents, args['output_file'])

            else:
                text_buffer.append(line)

    if len(text_buffer) > 0:
        documents.append(_create_document(text_buffer, parser))
    print('{} documents processed'.format(len(documents)))

    utils.pickle_to_file(documents, args['output_file'])
    print('All task finished')


if __name__ == '__main__':
    main()
