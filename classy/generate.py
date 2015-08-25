"""
.. py:module:: classy.generate
   :synopsis: Generate an inverted index to use in a text classifier.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
import pickle
import sys
from os import path
from classy.extract import Extractor, row_generator, row_generator_from_file
from numpy import zeros, diff, ones, cumsum, where
from classy.transform import Transformer, AnnotationTransformer, FeatureEncoder
from scipy.sparse import vstack, hstack, csc_matrix

L = logging.getLogger(__name__)


def generate_data(args):
    L.debug("%s", args)
    dialect = 'excel' if args.csv else 'plain'
    index = None
    vocabulary = None
    labels = None if args.no_label else []

    # LOAD VOCABULARY IF GIVEN

    if args.vocabulary and not args.replace and path.exists(args.vocabulary):
        L.info("loading existing vocabulary from '%s'", args.vocabulary)

        with open(args.vocabulary, 'rb') as f:
            vocabulary = pickle.load(f)

    # PARSE TEXT FILES AND BUILD INVERTED INDEX AND/OR VOCABULARY

    if not args.data:
        gen = row_generator(sys.stdin, dialect=dialect)
        index, vocabulary, labels = do(gen, args, vocabulary)
    else:
        grow = (vocabulary is None)

        for file in args.data:
            gen = row_generator_from_file(file, dialect=dialect,
                                          encoding=args.encoding)
            next_mat, next_vocab, next_lab = do(gen, args, vocabulary,
                                                grow=grow)

            if grow and vocabulary is not None:
                new_words = len(next_vocab) - len(vocabulary)

                if new_words > 0:
                    patch = zeros((index.shape[0], new_words))
                    index = hstack([index, patch])

            if index is None:
                index = next_mat
            else:
                index = vstack([index, next_mat])

            vocabulary = next_vocab

            if labels is not None:
                labels.extend(next_lab)

    index = index.tocsr()

    # REMOVE CUTOFF VOCABULARY

    if args.cutoff > 1 and (args.replace or not path.exists(args.vocabulary)):
        df = diff(csc_matrix(index, copy=False).indptr)  # document frequency
        mask = ones(len(df), dtype=bool)  # mask: columns that can/cannot stay
        mask &= df >= args.cutoff  # create a "mask" of columns above cutoff
        new_idx = cumsum(mask) - 1  # new indices (with array len as old)
        keep = where(mask)[0]  # determine which columns to keep
        index = index[:, keep]  # drop unused columns

        for word in list(vocabulary.keys()):
            idx = vocabulary[word]

            if mask[idx]:
                vocabulary[word] = new_idx[idx]
            else:
                del vocabulary[word]

    # WRITE VOCABULARY, LABELS, AND INVERTED INDEX

    L.info("index shape: %s vocabulary size: %s", index.shape, len(vocabulary))
    L.debug("vocabulary: %s", vocabulary.keys())

    if args.vocabulary and (args.replace or not path.exists(args.vocabulary)):
        L.info("writing vocabulary to '%s'", args.vocabulary)

        with open(args.vocabulary, 'wb') as f:
            pickle.dump(vocabulary, f)

    L.info("writing inverted index to '%s'", args.index)

    with open(args.index, 'wb') as f:
        pickle.dump([labels, index], f)


def do(generator, args, vocab, grow=False):
    stream = Extractor(generator, has_title=args.title,
                       lower=args.lowercase, decap=args.decap)
    stream = Transformer(stream, N=args.n_grams, K=args.k_shingles)

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
    return matrix, stream.vocabulary, stream.labels
