"""
.. py:module:: classy.transform
   :synopsis: Transform text and additional features into sparse matrices.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
import itertools
from array import array
from classy.etbase import Etc
from scipy.sparse import csr_matrix
from collections import defaultdict
from numpy import int32, ones, zeros

L = logging.getLogger(__name__)

# scikit-learn strategy - see
# https://github.com/scikit-learn/scikit-learn/blob/a95203b/sklearn/feature_extraction/text.py#L724


class Transformer(Etc):

    """
    Takes text from an Extractor that already has segmented and tokenized the
    text and converts the segmented tokens into a flat n-gram and k-shingle
    list.
    """

    def __init__(self, extractor, n=2, k=1):
        """
        :param extractor: the input Extractor stream
        :param n: the size of the n-grams to produce
                  (``1`` implies unigrams only)
        :param k: the size of the k-shingles to produce
                  (``1`` implies no shingling)
        """
        super(Transformer, self).__init__(extractor)
        L.debug("N=%s, K=%s", n, k)
        self.rows = iter(extractor)
        self.N = int(n)
        self.K = int(k)

        if self.N < 1:
            raise ValueError("N not a positive number")

        if self.K < 1:
            raise ValueError("K not a positive number")

    def __iter__(self):
        for row in self.rows:
            for i in self.text_columns:
                self._extract(row, i)

            yield row

    def _extract(self, row, i):
        n_grams = self.n_gram(row[i])
        shingles = self.k_shingle(row[i])
        row[i] = list(itertools.chain(n_grams, shingles))

    def n_gram(self, token_segments):
        """
        Yield consecutive n-grams from all ``token_segments``.
        The n-gram size, ``N``, is configured at instance level.

        :param token_segments: a list of a list of strings (tokens)
        :return: a n-gram generator; tokens are joined by space (`` ``).
        """
        ns = range(1, self.N + 1)

        for segment in token_segments:
            for n in ns:
                for i in range(len(segment) - n + 1):
                    yield " ".join(segment[i:i + n])

    def k_shingle(self, token_segments):
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
        L.debug("groups=%s, dropped_columns=%s", groups, dropped_columns)
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
            err = msg.format(col, col_name, self.text_columns)
            assert col in self.text_columns, err

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

    def __init__(self, transformer, vocabulary=None, grow_vocab=False,
                 id_col=0, label_col=-1):
        """
        :param transformer: the input Transformer stream
        :param vocabulary: optionally, use a predefined vocabulary
        :param id_col: the column containing the document/instance ID
                       (0-based; None implies there is no label column present)
        :param label_col: the column containing the document/instance label (0-
                          based; None implies there is no label column present)
        :param grow_vocab: expand the vocabulary (if given, instead of ignoring
                           missing words)
        """
        super(FeatureEncoder, self).__init__(transformer)
        L.debug("vocabulary=%s grow=%s id_col=%s label_col=%s",
                "None" if vocabulary is None else len(vocabulary),
                grow_vocab, id_col, label_col)
        self.rows = iter(transformer)
        self.id_col = None if id_col is None else int(id_col)
        self.label_col = None if label_col is None else int(label_col)
        self.vocabulary = None if vocabulary is None else dict(vocabulary)
        self.text_ids = []
        self.labels = []
        self._grow = grow_vocab and self.vocabulary is not None

    def _multirow_token_generator(self, row):
        template = '{}={}'

        for col in self.text_columns:
            name = self.names[col]

            for token in row[col]:
                yield template.format(name, token)

    def _unirow_token_generator(self, row):
        yield from row[self.text_columns[0]]

    def __iter__(self):
        vocab = self.vocabulary

        if vocab is None:
            raise AttributeError(
                "cannot stream-transform without a vocabulary"
            )

        n_features = len(vocab)

        if len(self.text_columns) == 1:
            token_generator = self._unirow_token_generator
        else:
            token_generator = self._multirow_token_generator

        for line, row in enumerate(self.rows):
            text_id = line + 1 if self.id_col is None else row[self.id_col]
            feature_counts = zeros(n_features, dtype=int32)

            for token in token_generator(row):
                try:
                    feature_counts[vocab[token]] += 1
                except KeyError:
                    pass  # ignore features not in the vocabulary

            yield text_id, feature_counts.T  # transpose to document array


    def make_sparse_matrix(self):
        indices = array('L')
        pointers = array('L')
        pointers.append(0)
        self.text_ids = []
        self.labels = []

        if self.vocabulary is None or self._grow:
            vocab = defaultdict(int)
            vocab.default_factory = vocab.__len__

            if self._grow and self.vocabulary is not None:
                vocab.update(self.vocabulary)
        else:
            vocab = self.vocabulary

        if len(self.text_columns) == 1:
            L.debug("generating tokens from column '%s'", self.names[self.text_columns[0]])
            token_generator = self._unirow_token_generator
        else:
            L.debug("generating tokens from columns: '%s'",
                    "', '".join(self.names[c] for c in self.text_columns))
            token_generator = self._multirow_token_generator

        for row in self.rows:
            if self.id_col is not None:
                self.text_ids.append(row[self.id_col])

            if self.label_col is not None:
                self.labels.append(row[self.label_col])

            for t in token_generator(row):
                try:
                    indices.append(vocab[t])
                except KeyError:
                    pass  # ignore features not in the vocabulary

            pointers.append(len(indices))

        if self.vocabulary is None or self._grow:
            self.vocabulary = dict(vocab)

        matrix = csr_matrix((ones(len(indices)), indices, pointers),
                            shape=(len(pointers) - 1, len(vocab)),
                            dtype=int32)
        matrix.sum_duplicates()
        return matrix
