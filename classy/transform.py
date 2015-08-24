"""
.. py:module::  classy.transform
   :synopsis: Transform text and additional features into sparse matrices.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

from array import array
from etbase import Etc
from scipy.sparse import csr_matrix
from collections import defaultdict
from numpy import int32, ones
import itertools


# Create sparse matrix with scipy:
# scipy.sparse.csr_matrix - see
# http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csr_matrix.html
# scikit-learn strategy - see
# https://github.com/scikit-learn/scikit-learn/blob/a95203b/sklearn/feature_extraction/text.py#L724

class Transformer(Etc):

    """
    Takes text from an Extractor that already has segmented and tokenized the
    text and converts the segmented tokens into a flat n-gram and k-shingle
    list.
    """

    def __init__(self, extractor, N=2, K=1):
        """
        :param extractor: the input Extractor stream
        :param N: the size of the n-grams to produce
                  (``1`` implies unigrams only)
        :param K: the size of the k-shingles to produce
                  (``1`` implies no shingling)
        """
        super(Transformer, self).__init__(extractor)
        self.rows = iter(extractor)
        self.N = int(N)
        self.K = int(K)

        if self.N < 1:
            raise ValueError("N not a positive number")

        if self.K < 1:
            raise ValueError("K not a positive number")

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
        """
        Yield consecutive n-grams from all ``token_segments``.
        The n-gram size, ``N``, is configured at instance level.

        :param token_segments: a list of a list of strings (tokens)
        :return: a n-gram generator; tokens are joined by space (`` ``).
        """
        N = range(1, self.N + 1)

        for segment in token_segments:
            for n in N:
                for i in range(len(segment) - n + 1):
                    yield " ".join(segment[i:i + n])

    def kshingle(self, token_segments):
        """
        Yield unique k-shingles by creating all possible combinations of
        unique words (tokens) in ``token_segments``.
        The k-shingle size, ``K``, is configured at instance level.
        Note that the order in which the words appeared in the text does not
        matter (unlike with n-grams).

        :param token_segments: a list of a list of strings (tokens)
        :return: a k-shingle generator; tokens are joined by underscore (``_``)
        """
        words = {w for s in token_segments for w in s if w.isalnum()}
        words = list(sorted(words))  # sorted to ensure uniqueness

        for k in range(2, self.K + 1):
            for shingle in itertools.combinations(words, k):
                yield "_".join(sorted(shingle))


class AnnotationTransformer(Etc):

    """
    Takes bag-of-word text arrays and attaches annotations to each item by
    prefixing them with reference defined in another column representing some
    external text annotation.

    Attachment is done by prefixing the text target column with the string in
    the annotation source column, separated by a colon character (``:``).
    """

    def __init__(self, transformer, groups, *dropped_columns):
        """
        :param transformer: the input Transformer stream
        :param groups: a dictionary where the keys are the text target column
                       and the values are the source annotation columns; for
                       example ``{1: (2, 3)}`` would annotate the second
                       (text) column with the annotations found in the third
                       and fourth columns (i.e., using 0-based column counts)
        :param dropped_columns: 0-based counts of columns to be dropped from
                                the stream after processing (e.g., to remove
                                the merged annotation columns)
        """
        super(AnnotationTransformer, self).__init__(transformer)
        self.rows = iter(transformer)
        self.dropped_cols = tuple(
            int(c) for c in sorted(set(dropped_columns), reverse=True)
        )
        self.groups = {
            int(token_col): tuple(int(c) for c in ann_cols)
            for token_col, ann_cols in groups.items()
        }
        names = list(self.names)

        for col in self.dropped_cols:
            try:
                del names[col]
            except IndexError as ex:
                msg = "names={}, dropped_columns={}; " \
                      "illegal dropped columns index?"
                err = msg.format(self.names, dropped_columns)
                raise RuntimeError(err) from ex

        if len(self.dropped_cols):
            self.names = names

        for col in self.groups:
            msg = "column {} [{}] not a known token column: {}"
            col_name = self._names[col] if col < len(self._names) else "ERROR"
            err = msg.format(col, col_name, self.token_columns)
            assert col in self.token_columns, err

    def __iter__(self):
        for row in self.rows:
            for token_col, annotation_cols in self.groups.items():
                try:
                    name = ':'.join(row[c] for c in annotation_cols)
                except IndexError as ex1:
                    msg = "len(row)={}, but annotation_col_indices={}"
                    raise RuntimeError(
                        msg.format(len(row), annotation_cols)
                    ) from ex1
                except TypeError as ex2:
                    msg = "not all annotation_columns={} are strings"
                    raise RuntimeError(msg.format(annotation_cols)) from ex2

                try:
                    row[token_col] = [
                        "{}:{}".format(name, token) for token in row[token_col]
                    ]
                except IndexError as ex3:
                    msg = "len(row)={}, but token_column_index={} [{}]"
                    raise RuntimeError(
                        msg.format(len(row), token_col, self._names[token_col])
                    ) from ex3

            for col in self.dropped_cols:
                try:
                    del row[col]
                except IndexError as ex4:
                    raise RuntimeError(
                        "row has no column {} to drop".format(col)
                    ) from ex4

            yield row


class FeatureEncoder(Etc):

    """
    Create an inverted index *and* a dictionary (vocabulary) from text columns,
    or create an inverted index using a given, predefined vocabulary.

    The inverted index is a sparse matrix of counts of each unique feature per
    instance.

    A vocabulary as produced by the encoder or as provided to it is a
    dictionary that maps unique feature strings to sparse matrix columns in the
    inverted index. If a provided vocabulary does not contain a feature found
    in the input, that missing feature is ignored (not counted).
    """

    def __init__(self, transformer, vocabulary=None, id_col=0, label_col=-1):
        """
        :param transformer: the input Transformer stream
        :param vocabulary: optionally, use a predefined vocabulary
        :param id_col: the column containing the document/instance ID
                       (0-based; None implies there is no label column present)
        :param label_col: the column containing the document/instance label (0-
                          based; None implies there is no label column present)
        """
        super(FeatureEncoder, self).__init__(transformer)
        self.rows = iter(transformer)
        self.id_col = None if id_col is None else int(id_col)
        self.label_col = None if label_col is None else int(label_col)
        self.vocabulary = None if vocabulary is None else dict(vocabulary)
        self.text_ids = []
        self.labels = []

    def _multirow_token_generator(self, row):
        template = '{}:{}'

        for col in self.token_columns:
            name = self.names[col]

            for token in row[col]:
                yield template.format(name, token)

    def _singlerow_token_generator(self, row):
        yield from row[self.token_columns[0]]

    def make_sparse_matrix(self):
        indices = array('L')
        indptr = array('L')
        indptr.append(0)
        self.text_ids = []
        self.labels = []

        if self.vocabulary is None:
            V = defaultdict(int)
            V.default_factory = V.__len__
        else:
            V = self.vocabulary

        if len(self.token_columns) == 1:
            token_generator = self._singlerow_token_generator
        else:
            token_generator = self._multirow_token_generator

        for row in self.rows:
            if self.id_col is not None:
                self.text_ids.append(row[self.id_col])

            if self.label_col is not None:
                self.labels.append(row[self.label_col])

            for t in token_generator(row):
                try:
                    indices.append(V[t])
                except KeyError:
                    pass  # ignore features not found in the vocabulary

            indptr.append(len(indices))

        if self.vocabulary is None:
            self.vocabulary = dict(V)

        matrix = csr_matrix((ones(len(indices)), indices, indptr),
                            shape=(len(indptr) - 1, len(V)),
                            dtype=int32)
        matrix.sum_duplicates()
        return matrix
