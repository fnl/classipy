"""
.. py:module:: classy.extract
   :synopsis: Extract column-based character text data.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
from csv import reader, register_dialect, QUOTE_NONE
from etbase import Etc
from segtok.segmenter import split_single
from segtok.tokenizer import word_tokenizer

L = logging.getLogger(__name__)
# noinspection PyArgumentList
register_dialect('plain', delimiter='\t', escapechar='\\', quoting=QUOTE_NONE)


def row_generator_from_file(path, encoding='utf-8', dialect='excel'):
    file_handle = open(path, 'rt', encoding=encoding)

    for row in row_generator(file_handle, dialect=dialect):
        yield row

    file_handle.close()


def row_generator(input_stream, dialect='plain'):
    yield from reader(input_stream, dialect)


class Extractor(Etc):

    """
    The extractor produces tokenized text(s) with any provided metadata.
    The data is stream should come from a row generator and the extractor
    in turn wraps the stream with a (one-off) iterator.

    Dataset attributes:

    - ``names`` contains the names (or 1-based number) of each column
    - ``text_columns`` are the (0-based) column numbers that contain text
    #- ``N`` are the max. n-grams to generate
    #- ``K`` are the max. k-shingles to generate; 0 produces nothing,
    #   1 generates all possible combinations of two tokens,
    #   2 *in addition* generates all combinations of bigrams,
    #   3 *in addition* generates all combos of trigrams, etc.
    """

    def __init__(self, row_gen, has_title=False, lower=False, decap=False):
        """
        :param row_gen: a row-generator yielding lists of columns
        :param has_title: whether the first row is the title row or not
        :param lower: whether to lower-case all tokens or not
        :param decap: whether to decapitalize the first character of each
                      sentence or not
        :return:
        """
        super(Extractor, self).__init__()
        text_columns = list()

        if lower:
            self._lower = lambda s: s.lower()
        else:
            self._lower = lambda s: s

        if decap:
            self._decap = lambda s: (
                '{}{}'.format(s[0].lower(), s[1:]) if s else s
            )
        else:
            self._decap = lambda s: s

        if has_title:
            self.names = next(row_gen)
            self._curr_row = next(row_gen)
        else:
            self._curr_row = next(row_gen)
            self.names = (str(i + 1) for i in range(len(self._curr_row)))

        self._row_len = len(self._curr_row)
        self._row_gen = row_gen

        for column_nr, cell_content in enumerate(self._curr_row):
            if ' ' in cell_content or len(cell_content) == 0:
                # noinspection PyUnresolvedReferences
                text_columns.append(column_nr)

        self.text_columns = text_columns
        L.debug('text columns: %s', ' '.join(
            self.names[i] for i in self.text_columns
        ))

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_row is None:
            raise StopIteration()

        row = self._curr_row

        if len(row) != self._row_len:
            raise IOError('found %d, expected %d columns at line %d:\n%s' % (
                len(row), self._row_len, self._row_gen.line_num, str(row)
            ))

        try:
            self._curr_row = next(self._row_gen)
        except StopIteration:
            self._curr_row = None

        data = tuple((row[col] if col not in self.text_columns else [])
                     for col in range(self._row_len))

        for col in self.text_columns:
            for sentence in split_single(row[col]):
                sentence = self._decap(sentence)
                tokens = [self._lower(t) for t in word_tokenizer(sentence)]
                data[col].append(tokens)

        return data
