"""
TODO: Description
"""
import sys
from extract import Extractor, row_generator, row_generator_from_file
import os
from transform import Transformer, AnnotationTransformer, FeatureEncoder

__author__ = 'Florian Leitner <florian.leitner@gmail.com>'

def generate_data(args):
    dialect = 'excel' if args.csv else 'plain'
    if not args.data:
        gen = row_generator(sys.stdin, dialect=dialect)
        mat, voc = do(gen, args)
    else:
        for file in args.data:
            gen = row_generator_from_file(file, dialect=dialect,
                                          encoding=args.encoding)
            mat, voc = do(gen, args)

def do(generator, args):
    stream = Extractor(generator, has_title=args.title,
                       lower=args.lowercase, decap=args.decap)
    stream = Transformer(stream, N=args.n_grams, K=args.k_shingles)

    if args.annotate:
        groups = {i: args.annotate for i in stream.token_columns}
        stream = AnnotationTransformer(stream, groups, *args.annotate)

    label_col = None if args.no_label else -1

    if args.label_first:
        label_col = 0
    elif args.label_second:
        label_col = 1

    id_col = None if args.no_id else 0

    if id_col == 0 and label_col == 0:
        id_col = 1

    vocabulary = None

    if args.vocabulary and os.path.exists(args.vocabulary):
        with open(args.vocabulary, 'rb') as f:
            vocabulary = args.vocabulary.load(f)

    stream = FeatureEncoder(stream, vocabulary=vocabulary,
                            id_col=id_col, label_col=label_col)
    matrix = stream.make_sparse_matrix()
    return matrix, stream.vocabulary