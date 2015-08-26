"""
.. py:module:: classy.generate
   :synopsis: Generate an inverted index to use in a text classifier.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
import pickle
import sys
from classy.data import save_index, save_vocabulary
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
    text_ids = None if args.no_id else []

    if args.annotate:
        args.annotate = [a - 1 for a in args.annotate]

        if any(a < 0 for a in args.annotate):
            raise ValueError("not all annotate columns are positive integers")

    # LOAD VOCABULARY IF GIVEN

    if args.vocabulary and not args.replace and path.exists(args.vocabulary):
        L.info("loading existing vocabulary from '%s'", args.vocabulary)

        with open(args.vocabulary, 'rb') as f:
            vocabulary = pickle.load(f)

    # PARSE TEXT FILES AND BUILD INVERTED INDEX AND/OR VOCABULARY

    if not args.data:
        gen = row_generator(sys.stdin, dialect=dialect)
        index, vocabulary, labels, text_ids = do(gen, args, vocabulary)
    else:
        grow = (vocabulary is None)

        for file in args.data:
            gen = row_generator_from_file(file, dialect=dialect,
                                          encoding=args.encoding)
            next_mat, next_vocab, next_lab, next_ids = do(
                gen, args, vocabulary, grow=grow
            )

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

            if text_ids is not None:
                text_ids.extend(next_ids)

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

    save_index(text_ids, labels, index, args.index)

    if args.vocabulary and (args.replace or not path.exists(args.vocabulary)):
        save_vocabulary(vocabulary, index, args.vocabulary)


def do(generator, args, vocab, grow=False):
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
