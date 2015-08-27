"""
.. py:module:: classy.predict
   :synopsis: Predict labels for an inverted index.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
import sys
from classy.data import load_index, get_n_rows, load_vocabulary
from classy.extract import row_generator, row_generator_from_file, Extractor
from classy.transform import Transformer, AnnotationTransformer, FeatureEncoder
from sklearn.externals import joblib

L = logging.getLogger(__name__)


def predict_labels(args):
    L.debug("%s", args)

    if args.text:
        stream_predictor(args)
    else:
        batch_predictor(args)


def batch_predictor(args):
    data = load_index(args.index[0])

    if data.labels is None or len(data.labels) == 0:
        raise RuntimeError("input data has no labels to learn from")

    pipeline = joblib.load(args.model)
    predictions = pipeline.predict(data.index)

    if data.text_ids:
        text_ids = data.text_ids
    else:
        text_ids = range(1, get_n_rows(data) + 1)

    for text_id, prediction in zip(text_ids, predictions):
        print(text_id, data.label_names[prediction], sep='\t')


def stream_predictor(args):
    if args.vocabulary is None:
        args.error('missing required option for text input: --vocabulary VOCAB')

    vocabulary = load_vocabulary(args.vocabulary)
    dialect = 'excel' if args.csv else 'plain'

    if args.annotate:  # transform to zero-based count
        args.annotate = [a - 1 for a in args.annotate]

        if any(a < 0 for a in args.annotate):
            raise ValueError("not all annotate columns are positive integers")

    if not args.index:
        gen = row_generator(sys.stdin, dialect=dialect)
        predict_from(gen, args, vocabulary)
    else:
        for file in args.index:
            gen = row_generator_from_file(file, dialect=dialect,
                                          encoding=args.encoding)
            predict_from(gen, args, vocabulary)


def predict_from(text, args, vocab):
    stream = Extractor(text, has_title=args.title,
                       lower=args.lowercase, decap=args.decap)
    stream = Transformer(stream, n=args.n_grams, k=args.k_shingles)

    if args.annotate:
        groups = {i: args.annotate for i in stream.text_columns}
        stream = AnnotationTransformer(stream, groups, *args.annotate)

    if args.no_id:
        id_col = None
    elif args.id_second:
        id_col = 1
    elif args.id_last:
        id_col = -1
    else:
        id_col = 0

    stream = FeatureEncoder(stream, vocabulary=vocab,
                            id_col=id_col, label_col=None)
    pipeline = joblib.load(args.model)
    make_label = str

    if args.label:
        n_labels = len(args.label)
        make_label = lambda i: args.label[i] if i < n_labels else i

    for text_id, features in stream:
        prediction = pipeline.predict(features)[0]
        print(text_id, make_label(prediction), sep='\t')
