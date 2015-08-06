"""
.. py:module::  classy.transform
   :synopsis: Transform text and additional features into sparse matrices.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""
import logging
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
        self._token_cols = []

    def __iter__(self):
        self._token_cols = []
        row = next(self.rows)

        for i, v in enumerate(row):
            if isinstance(v, list):
                self._token_cols.append(i)
                self._extract(row, i)

        if len(self._token_cols) == 0:
            logging.warning('Transformer found no token columns in input')

        yield row

        for row in self.rows:
            for i in self._token_cols:
                self._extract(row, i)

            yield row

    def _extract(self, row, i):
        if row[i] and len(row[i]):
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

    def __iter__(self):
        row = next(self.rows)

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
                assert isinstance(row[token_col], list), "column={} not a list".format(token_col)
                row[token_col] = ['{}:{}'.format(name, token) for token in row[token_col]]
            except IndexError as ex3:
                msg = "len(row)={}, but token_column_index={}"
                raise RuntimeError(msg.format(len(row), token_col)) from ex3

        for col in self.dropped_cols:
            try:
                del row[col]
            except IndexError as ex4:
                raise RuntimeError('row has no column {} to drop'.format(col)) from ex4

        yield row


class FeatureEncoder:
    def __init__(self, transformer):
        self.rows = transformer
        self.feature_cols = None

    def __iter__(self):
        return self

    def __next__(self):
        row = next(self.rows)

        if self.feature_cols is None:
            self.feature_cols = []
            found_text = False

            for i, col in row:
                if isinstance(col, list):
                    found_text = True
                elif found_text:
                    self.feature_cols.append((i, []))

        for i, features in self.feature_cols:
            idx = features.index(row[i])

            if idx == -1:
                idx = len(features)
                features.append(row[i])
