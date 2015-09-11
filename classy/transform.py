"""
.. py:module:: classy.transform
   :synopsis: Transform text and its annotations into a sparse matrix.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
import itertools
from array import array
from classy.etbase import Etc
from classy.extract import Extractor
from scipy.sparse import csr_matrix
from collections import defaultdict
from numpy import int32, ones, zeros

L = logging.getLogger(__name__)


class NGramTransformer(Etc):

    """
    Takes text from an Extractor that already has segmented and tokenized the
    text and converts the segmented tokens into a flat n-gram list.
    """

    def __init__(self, extractor, n=1):
        """
        :param extractor: the input Extractor stream
        :param n: the size of the n-grams to produce
                  (``1`` implies unigrams only)
        """
        super(NGramTransformer, self).__init__(extractor)
        L.debug("N=%s", n)
        self.rows = iter(extractor)
        self.N = int(n)

        if self.N < 1:
            raise ValueError("N not a positive number")

    def __iter__(self):
        for row in self.rows:
            for i in self.text_columns:
                self._extract(row, i)

            yield row

    def _extract(self, row, i):
        if len(row[i]) == 0:  # mark void fields as such
            row[i] = ['~void~']
        else:
            row[i] = list(self.n_gram(row[i]))

    def n_gram(self, token_segments):
        """
        Yield consecutive n-grams from all ``token_segments``.
        The n-gram size, ``N``, is configured at instance level.

        :param token_segments: a list of a list of strings (tokens)
        :return: a n-gram generator; tokens are joined by ``+``.
        """
        ns = range(1, self.N + 1)

        for segment in token_segments:
            for n in ns:
                for i in range(len(segment) - n + 1):
                    if n == 1 or all(t.isalnum() for t in segment[i:i + n]):
                        yield "+".join(segment[i:i + n])


class KShingleTransformer(Etc):

    """
    Takes text from a Transformer where every text field is already in the
    form of a list of tokens and converts it into a k-shingle list (i.e.,
    generating all k-combinations of tokens).

    Should be used *after* an AnnotationTransformer to join tokens and
    annotations.
    """

    def __init__(self, extractor, k=1):
        """
        :param extractor: the input Extractor stream
        :param k: the size of the k-shingles to produce
                  (``1`` implies no shingling)
        """
        super(KShingleTransformer, self).__init__(extractor)
        L.debug("K=%s", k)
        self.rows = iter(extractor)
        self.K = int(k)

        if self.K < 1:
            raise ValueError("K not a positive number")

    def __iter__(self):
        for row in self.rows:
            if self.K > 1:
                for i in self.text_columns:
                    row[i].extend(self.k_shingle(row[i]))

            yield row

    def k_shingle(self, tokens):
        """
        Yield unique k-shingles by creating all possible combinations of
        unique words (tokens) in ``tokens``.
        The k-shingle size, ``K``, is configured at instance level.
        Note that the order in which the words appeared in the text does not
        matter (unlike with n-grams).

        :param tokens: the list of tokens (strings)
        :return: a k-shingle generator; tokens are joined by ``_``
        """
        words = {t for t in tokens if t.isalnum()}
        words = list(sorted(words))  # sorted to ensure uniqueness

        for k in range(2, self.K + 1):
            for shingle in itertools.combinations(words, k):
                yield "_".join(sorted(shingle))


class AnnotationTransformer(Etc):

    """
    Takes a transformer and *attaches* annotations to each *token* by
    prefixing them with that annotation.

    Attachment is done by prefixing the text target column with the name of
    the annotation column followed by a hash (``#``), the string in the
    annotation source column, and separated from the token by a colon
    character (``:``).
    """

    def __init__(self, transformer, groups):
        """
        :param transformer: the input Transformer stream
        :param groups: a dictionary where the keys are the text target column
                       and the values are the source annotation columns; for
                       example ``{1: (2, 3)}`` would annotate the second
                       (text) column with the annotations found in the third
                       and fourth columns (i.e., using 0-based column counts)
        """
        super(AnnotationTransformer, self).__init__(transformer)
        L.debug("groups=%s", groups)
        self.rows = iter(transformer)
        self.groups = {
            int(token_col): tuple(int(c) for c in ann_cols)
            for token_col, ann_cols in groups.items()
        }

        for col in self.groups:
            msg = "column {} [{}] not a known token column: {}"
            col_name = self._names[col] if col < len(self._names) else "ERROR"
            err = msg.format(col, col_name, self.text_columns)
            assert col in self.text_columns, err
            sources = [self.names[i] if len(self.names) > i else "ERROR"
                       for i in self.groups[col]]
            L.debug("group %s <= %s", self.names[col], sources)

    def __iter__(self):
        for row in self.rows:
            for token_col, annotation_cols in self.groups.items():
                try:
                    annotations = ['{:s}#{:s}'.format(self.names[c], row[c])
                                   for c in annotation_cols]
                except IndexError as ex1:
                    msg = "len(row)={}, but annotation_col_indices={}"
                    raise RuntimeError(
                        msg.format(len(row), annotation_cols)
                    ) from ex1
                except TypeError as ex2:
                    msg = "not all annotation_columns={} are strings: {}"
                    raise RuntimeError(
                        msg.format(annotation_cols,
                                   [type(row[c]) for c in annotation_cols])
                    ) from ex2

                try:
                    ann_tokens = []

                    for name in annotations:
                        ann_tokens.extend("{:s}:{:s}".format(name, token) for
                                          token in row[token_col])

                    row[token_col].extend(ann_tokens)
                except IndexError as ex3:
                    msg = "len(row)={}, but token_column_index={} [{}]"
                    raise RuntimeError(
                        msg.format(len(row), token_col, self._names[token_col])
                    ) from ex3

            yield row


class FeatureTransformer(Etc):

    """
    Takes a transformer and *appends* annotations as new features to each text
    field.

    Appending is done by adding the a string consisting of the name of the
    annotation column and the annotation itself, separated by a colon
    character (``:``).
    """

    def __init__(self, transformer, columns, binarize=False):
        """
        :param transformer: the input Transformer stream
        :param columns: weave annotation columns into the token lists
                        as additional "tokens"; (0-based)
        :param binarize: also add all binary combinations of the annotations
        """
        super(FeatureTransformer, self).__init__(transformer)
        L.debug("feature columns=%s%s", columns, " BINARY" if binarize else "")
        self.rows = iter(transformer)
        self.feature_columns = list(int(c) for c in columns)
        self.binarize = binarize
        L.debug("features: %s", [self.names[i] for i in self.feature_columns])

    def __iter__(self):
        names = self.names
        cols = self.feature_columns

        for row in self.rows:
            if cols:
                for txt in self.text_columns:
                    tokens = row[txt]
                    feats = [
                        "{:s}#{:s}".format(names[c], row[c]) for c in cols
                    ]
                    tokens.extend(feats)

                    if self.binarize:
                        for pair in itertools.combinations(feats, 2):
                            tokens.append("+".join(pair))

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
        template = '{:s}={:s}'

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

            if line % 1000:
                L.info("processed %s documents", line)

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
            L.debug("generating tokens from column '%s'",
                    self.names[self.text_columns[0]])
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

            if len(pointers) % 1000 == 0:
                L.info("processed %s documents (vocab. size: %s)",
                       len(pointers), len(vocab))

        if self.vocabulary is None or self._grow:
            self.vocabulary = dict(vocab)

        matrix = csr_matrix((ones(len(indices)), indices, pointers),
                            shape=(len(pointers) - 1, len(vocab)),
                            dtype=int32)
        matrix.sum_duplicates()
        return matrix


def transform_input(generator, args):
    stream = Extractor(generator, has_title=args.title,
                       lower=args.lowercase, decap=args.decap)
    stream = NGramTransformer(stream, n=args.n_grams)

    if args.k_shingles > 1:
        stream = KShingleTransformer(stream, k=args.k_shingles)

    if args.annotate:
        groups = {i: args.annotate for i in stream.text_columns}
        stream = AnnotationTransformer(stream, groups)

    if args.feature:
        stream = FeatureTransformer(stream, args.feature, args.binarize)

    return stream
