"""
.. py:module::  classy.transform
   :synopsis: Transform text and additional features into sparse matrices.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""
from array import array
from scipy.sparse import csr_matrix
from collections import defaultdict
from numpy import int32
import itertools


# Create sparse matrix with scipy:
# scipy.sparse.csr_matrix - see
# http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csr_matrix.html
# scikit-learn strategy - see
# https://github.com/scikit-learn/scikit-learn/blob/a95203b/sklearn/feature_extraction/text.py#L724

class Transformer:
    def __init__(self, extractor, N=2, K=1):
        self.rows = iter(extractor)
        self.N = N
        self.K = K
        self.token_columns = extractor.text_columns
        self.names = extractor.names

    def __iter__(self):
        for row in self.rows:
            for i in self.token_columns:
                self._extract(row, i)

            yield row

    def _extract(self, row, i):
        ngrams = self.ngram(row[i])
        shingles = self.kshingle(row[i])
        row[i] = list(itertools.chain(ngrams, shingles))

    def ngram(self, token_segments):
        N = range(1, self.N + 1)

        for segment in token_segments:
            for n in N:
                for i in range(len(segment) - n + 1):
                    yield " ".join(segment[i:i + n])

    def kshingle(self, token_segments):
        words = sorted(set(w for s in token_segments for w in s if w.isalnum()))
        words = list(words)

        for k in range(2, self.K + 1):
            for shingle in itertools.combinations(words, k):
                yield "_".join(sorted(shingle))


class AnnotationTransformer:
    def __init__(self, transformer, groups, *dropped_columns):
        self.rows = iter(transformer)
        self.dropped_cols = tuple(int(c) for c in sorted(set(dropped_columns), reverse=True))
        self.groups = {
            int(token_col): tuple(int(c) for c in ann_cols)
            for token_col, ann_cols in groups.items()
        }
        self.token_columns = transformer.token_columns
        self._names = transformer.names
        self.names = list(self._names)

        for col in self.dropped_cols:
            try:
                del self.names[col]
            except IndexError as ex:
                msg = "names={}, dropped_columns={}; illegal dropped columns index?"
                raise RuntimeError(msg.format(self._names, dropped_columns)) from ex


        for col in self.groups:
            msg = "column {} [{}] not a known token column: {}"
            assert col in self.token_columns, \
                msg.format(col, self._names[col] if col < len(self._names) else "ERROR",
                           self.token_columns)

    def __iter__(self):
        for row in self.rows:
            for token_col, annotation_cols in self.groups.items():
                try:
                    name = ':'.join(row[c] for c in annotation_cols)
                except IndexError as ex1:
                    msg = "len(row)={}, but annotation_col_indices={}"
                    raise RuntimeError(msg.format(len(row), annotation_cols)) from ex1
                except TypeError as ex2:
                    msg = "not all annotation_columns={} are strings"
                    raise RuntimeError(msg.format(annotation_cols)) from ex2

                try:
                    row[token_col] = ['{}:{}'.format(name, token) for token in row[token_col]]
                except IndexError as ex3:
                    msg = "len(row)={}, but token_column_index={} [{}]"
                    raise RuntimeError(
                        msg.format(len(row), token_col, self._names[token_col])
                    ) from ex3

            for col in self.dropped_cols:
                try:
                    del row[col]
                except IndexError as ex4:
                    raise RuntimeError('row has no column {} to drop'.format(col)) from ex4

            yield row


def ones(num):
    for i in range(num):
        yield 1


class FeatureEncoder:
    def __init__(self, transformer, vocabulary=None, id_col=0, label_col=-1):
        self.rows = transformer
        self.token_columns = transformer.token_columns
        self.names = transformer.names
        self.id_col = id_col
        self.label_col = label_col
        self.vocabulary = vocabulary
        self.text_ids = []
        self.labels = []

    def _multirow_token_generator(self, row):
        template = '{}={}'

        for col in self.token_columns:
            name = self.names[col]

            for token in row[col]:
                yield template.format(name, token)

    def _singlerow_token_generator(self, row):
        yield from row[self.token_columns[0]]

    def make_sparse_matrix(self):
        indices = array('L')
        indptr = array('L')
        self.text_ids = []
        self.labels = []

        if self.vocabulary is None:
            V = defaultdict(int)
            V.default_factory = self.vocabulary.__len__
        else:
            V = self.vocabulary

        if len(self.token_columns) == 1:
            token_generator = self._singlerow_token_generator
        else:
            token_generator = self._multirow_token_generator

        for row in self.rows:
            self.text_ids.append(row[self.id_col])
            self.labels.append(row[self.label_col])

            for t in token_generator(row):
                try:
                    indices.append(V[t])
                except KeyError:
                    pass # ignore missing vocabulary

            indptr.append(len(indices))

        if self.vocabulary is None:
            self.vocabulary = dict(V)

        matrix = csr_matrix((ones(len(indices)), indices, indptr),
                            shape=(len(indptr) - 1, len(V)),
                            dtype=int32)
        matrix.sum_duplicates()
        return matrix
