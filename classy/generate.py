"""
.. py:module:: classy.generate
   :synopsis: Generate an inverted index to use in a text classifier.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
import sys
from os import path
from numpy import zeros
from scipy.sparse import vstack, hstack
from .data import save_index, save_vocabulary, make_data, load_vocabulary
from .extract import Extractor, row_generator, row_generator_from_file
from .transform import NGramTransformer, AnnotationTransformer, FeatureEncoder, KShingleTransformer, \
    FeatureTransformer, transform_input
from .select import drop_words, select_best, eliminate_words


L = logging.getLogger(__name__)


def generate_data(args):
    """
    Command: generate

    :param args: command-line arguments
    :raise ValueError: illegal --annotate column indexes
    """
    L.debug("%s", args)

    if args.vocabulary and not args.replace and path.exists(args.vocabulary):
        vocabulary = load_vocabulary(args.vocabulary)
    else:
        vocabulary = None

    args.annotate = fix_column_offset(args.annotate)
    args.feature = fix_column_offset(args.feature)

    if not args.data:
        data, vocabulary = parse_stdin(args, vocabulary)
    else:
        data, vocabulary = parse_files(args, vocabulary)

    if args.cutoff > 1:
        data = drop_words(args.cutoff, data, vocabulary)

    if args.select > 0:
        data = select_best(args.select, data, vocabulary)

    if args.eliminate > 0:
        data = eliminate_words(args.eliminate, data, vocabulary)

    save_index(data, args.index)

    if args.vocabulary and (args.replace or not path.exists(args.vocabulary)):
        save_vocabulary(vocabulary, data, args.vocabulary)


def fix_column_offset(columns):
    """Transform one-based column indices to zero-based indices."""
    if columns:
        columns = [c - 1 for c in columns]

        if any(c < 0 for c in columns):
            raise ValueError("not all selected columns are positive integers")

    return columns


def parse_stdin(args, vocabulary):
    """
    Keep reading from STDIN until the stream ends,
    building the index and vocabulary.

    :param args: command-line arguments
    :param vocabulary: a vocabulary dictionary
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

    :param args: command-line arguments
    :param vocabulary: a vocabulary dictionary
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
    stream = transform_input(generator, args)
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


