"""Wrapper class for nltk Starford Parser with lexicalized output."""

from nltk.parse.stanford import StanfordParser


class LexicalizedStanfordParser(StanfordParser):
    """Wrapper class for StanfordParser with lexicalized output."""

    _OUTPUT_FORMAT_OPTIONS = 'lexicalize'

    def parse_sents(self, sentences, verbose=False):
        """
        Use StanfordParser to parse multiple sentences. Takes multiple sentences as a
        list where each sentence is a list of words.
        Each sentence will be automatically tagged with this StanfordParser instance's
        tagger.
        If whitespaces exists inside a token, then the token will be treated as
        separate tokens.

        :param sentences: Input sentences to parse
        :type sentences: list(list(str))
        :rtype: iter(iter(Tree))
        """
        cmd = [
            self._MAIN_CLASS,
            '-model', self.model_path,
            '-sentences', 'newline',
            '-outputFormat', self._OUTPUT_FORMAT,
            '-outputFormatOptions', self._OUTPUT_FORMAT_OPTIONS,
            '-tokenized',
            '-escaper', 'edu.stanford.nlp.process.PTBEscapingProcessor',
        ]
        return self._parse_trees_output(self._execute(
            cmd, '\n'.join(' '.join(sentence) for sentence in sentences), verbose))

    def raw_parse_sents(self, sentences, verbose=False):
        """
        Use StanfordParser to parse multiple sentences. Takes multiple sentences as a
        list of strings.
        Each sentence will be automatically tokenized and tagged by the Stanford Parser.

        :param sentences: Input sentences to parse
        :type sentences: list(str)
        :rtype: iter(iter(Tree))
        """
        print 'lalalal'
        print self._OUTPUT_FORMAT_OPTIONS
        cmd = [
            self._MAIN_CLASS,
            '-model', self.model_path,
            '-sentences', 'newline',
            '-outputFormat', self._OUTPUT_FORMAT,
            '-outputFormatOptions', self._OUTPUT_FORMAT_OPTIONS,
        ]
        return self._parse_trees_output(self._execute(cmd, '\n'.join(sentences), verbose))
