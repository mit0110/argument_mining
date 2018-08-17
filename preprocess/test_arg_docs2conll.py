try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import unittest

from arg_docs2conll import AnnotatedJudgementFactory
from build_conll import DocumentWriter


class AnnotatedJudgementFactoryTests(unittest.TestCase):

    def test_annotation(self):
        FILENAME = 'test_files/CASE_OF__TALMANE_v._LATVIA.txt'
        EXPECTED_FILENAME = 'test_files/expected.conll'
        with AnnotatedJudgementFactory(FILENAME) as instance_extractor:
            document = instance_extractor.build_document()
        output_file = StringIO()
        writer = DocumentWriter(output_file, include_relations=False,
                                separation='paragraph')
        writer.write_document(document)
        with open(EXPECTED_FILENAME, 'r') as f:
            expected_file = f.read()
        for line_number, (output_line, expected_line) in enumerate(zip(
            output_file.getvalue().split('\n'), expected_file.split('\n'))):
            self.assertEqual(
                expected_line, output_line,
                msg="Failure in line {}, {} != {}".format(
                    line_number, expected_line, output_line))


if __name__ == '__main__':
    unittest.main()