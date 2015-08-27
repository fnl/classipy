"""
.. py:module:: classy.generate
   :synopsis: Generate an inverted index to use in a text classifier.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
import sys
from classy.data import save_index, save_vocabulary, make_data, load_vocabulary
from os import path
from classy.extract import Extractor, row_generator, row_generator_from_file
from numpy import zeros, diff, ones, cumsum, where
from classy.transform import Transformer, AnnotationTransformer, FeatureEncoder
from scipy.sparse import vstack, hstack, csc_matrix

L = logging.getLogger(__name__)


def generate_data(args):
    L.debug("%s", args)

    if args.vocabulary and not args.replace and path.exists(args.vocabulary):
        vocabulary = load_vocabulary(args.vocabulary)
    else:
        vocabulary = None

    if args.annotate:  # transform to zero-based count
        args.annotate = [a - 1 for a in args.annotate]

        if any(a < 0 for a in args.annotate):
            raise ValueError("not all annotate columns are positive integers")

    if not args.data:
        data, vocabulary = parse_stdin(args, vocabulary)
    else:
        data, vocabulary = parse_files(args, vocabulary)

    if args.cutoff > 1 and (args.replace or not path.exists(args.vocabulary)):
        data = drop_words(args.cutoff, data, vocabulary)

    save_index(data, args.index)

    if args.vocabulary and (args.replace or not path.exists(args.vocabulary)):
        save_vocabulary(vocabulary, data, args.vocabulary)


def parse_stdin(args, vocabulary):
    """
    Keep reading from STDIN until the stream ends,
    building the index and vocabulary.

    :return: a new Data structure and the vocabulary
    """
    dialect = 'excel' if args.csv else 'plain'
    gen = row_generator(sys.stdin, dialect=dialect)
    index, vocabulary, labels, text_ids = _do(gen, args, vocabulary)
    return make_data(index, text_ids, labels), vocabulary


def parse_files(args, vocabulary):
    """
    Build an index, possibly from multiple files
    via iterative expansions of the index and vocabulary per file.

    :return: a new Data structure and the vocabulary
    """
    dialect = 'excel' if args.csv else 'plain'
    labels = None if args.no_label else []
    text_ids = None if args.no_id else []
    grow = (vocabulary is None)
    index = None

    for file in args.data:
        gen = row_generator_from_file(file, dialect=dialect,
                                      encoding=args.encoding)
        next_mat, next_vocab, next_lab, next_ids = _do(
            gen, args, vocabulary, grow=grow
        )

        if grow and vocabulary is not None:  # false on the first round
            # (i.e., when index still is None!)
            new_words = len(next_vocab) - len(vocabulary)

            if new_words > 0:
                # add new columns for the new vocabulary words
                patch = zeros((index.shape[0], new_words))
                index = hstack([index, patch])

        if index is None:
            index = next_mat
        else:
            # now its safe to "paste" the new rows below the current rows
            index = vstack([index, next_mat])

        vocabulary = next_vocab

        if labels is not None:
            labels.extend(next_lab)

        if text_ids is not None:
            text_ids.extend(next_ids)

    return make_data(index, text_ids, labels), vocabulary


def _do(generator, args, vocab=None, grow=False):
    """
    Run the actual extraction/transformation from an input stream.

    :param generator: the input stream/row generator
    :param args: the command line arguments
    :param vocab: the (current) vocabulary (if any)
    :param grow: whether to expand the vocabulary or drop words no in it;
                 only has an effect if there is an actual vocabulary given
    :return: the inverted index, vocabulary, labels, and text IDs
    """
    stream = Extractor(generator, has_title=args.title,
                       lower=args.lowercase, decap=args.decap)
    stream = Transformer(stream, n=args.n_grams, k=args.k_shingles)

    if args.annotate:
        groups = {i: args.annotate for i in stream.text_columns}
        stream = AnnotationTransformer(stream, groups, *args.annotate)

    label_col = None if args.no_label else -1

    if args.label_first:
        label_col = 0
    elif args.label_second:
        label_col = 1

    id_col = None if args.no_id else 0

    if id_col == 0 and label_col == 0:
        id_col = 1

    stream = FeatureEncoder(stream, vocabulary=vocab, grow_vocab=grow,
                            id_col=id_col, label_col=label_col)
    matrix = stream.make_sparse_matrix()
    labels = stream.labels if stream.labels else None
    text_ids = stream.text_ids if stream.text_ids else None
    return matrix, stream.vocabulary, labels, text_ids


def drop_words(min_df, data, vocabulary):
    """
    Prune words below some minimum document frequency ``min_df`` from the
    vocabulary (in-place) and drop those columns from the inverted index.

    :return: a new Data structure
    """
    df = diff(csc_matrix(data.index, copy=False).indptr)  # document frequency
    mask = ones(len(df), dtype=bool)  # mask: columns that can/cannot stay
    mask &= df >= min_df  # create a "mask" of columns above cutoff
    new_idx = cumsum(mask) - 1  # new indices (with array len as old)
    keep = where(mask)[0]  # determine which columns to keep
    data = data._replace(index=data.index[:, keep])  # drop unused columns

    for word in list(vocabulary.keys()):
        idx = vocabulary[word]

        if mask[idx]:
            vocabulary[word] = new_idx[idx]
        else:
            del vocabulary[word]

    return data
