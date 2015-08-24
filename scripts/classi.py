#!/usr/bin/env python3

"""A command-line tool for text classification."""

import argparse
from extract import row_generator
import os.path
import re
import sys
import warnings

__author__ = "Florian Leitner <florian.leitner@gmail.com>"
__verison__ = "1.0"

def generate_data(args):
    if not args.data:
        stream = row_generator(sys.stdin,
                               dialect='excel' if args.csv else 'plain')
    else:

    print("generate data")

def learn_model(args):
    print("learn model")

def evaluate_model(args):
    print("evaluate model")

def predict_labels(args):
    print("predict labels")

# Program Setup
# =============

parser = argparse.ArgumentParser(
    description="A command-line tool for text classification."#,
    #usage="%(prog)s COMMAND [OPTIONS] [FILE...]"
)

# parser.add_argument("command", metavar='COMMAND', choices=["generate", "learn", "evaluate", "predict"])
commands = parser.add_subparsers(help='a command (use `CMD -h` to get help about a command):', metavar="CMD")
generate = commands.add_parser('generate', help='an inverted index from text input and/using a vocabulary', aliases=['g', 'ge', 'gen', 'gene', 'gener', 'genera', 'generat'])
learn = commands.add_parser('learn', help='a model from inverted index data', aliases=['l', 'le', 'lea', 'lear'])
evaluate = commands.add_parser('evaluate', help='by learing from an inverted index', aliases=['e', 'ev', 'eva', 'eval', 'evalu', 'evalua', 'evaluat'])
predict = commands.add_parser('predict', help='unlabeled data using a learned model', aliases=['p', 'pr', 'pre', 'pred', 'predi', 'predic'])

# parser.add_argument("data", metavar='FILE', nargs='*', help="input text file(s) (generate can read from STDIN, too) or inverted index matrix")

# featgen = parser.add_argument_group('GENERATE options for feature generation from TSV input')
generate.add_argument('data', metavar='FILE', nargs='*', help="input text file(s) (default: read from STDIN)")
generate.add_argument("--csv", action='store_true', help='input is Excel-like CSV with double-quoted escapes and string values in quotes (default: TSV without quotes)')
generate.add_argument("--title", action='store_true', help='input has a title row (default: no title row, use numbers instead)')
generate.add_argument("--decap", action='store_true', help="lowercase the first letter of each sentence")
generate.add_argument("--lowercase", action='store_true', help="lowercase all letters")
generate.add_argument("--encoding", type=str, default='utf-8', help="encoding of input files (default: UTF-8)")
generate.set_defaults(func=generate_data)
generate.add_argument("--n-grams", metavar='N', type=int, default=1, help="generate N-grams of all words in a sentence")
generate.add_argument("--k-shingles", metavar='K', type=int, default=1, help="generate all combinations of any K n-grams in the instance")
generate.add_argument("--annotate", metavar='COL', type=int, action='append', help="annotate text columns with the extra label from this/these column(s)")
generate.add_argument("--vocabulary", metavar='FILE', type=str, help="vocabulary file (will be generated if absent)")
generate.add_argument("--label-first", action='store_true', help='the nominal class label is the first instead of the last column')
generate.add_argument("--label-second", action='store_true', help='the nominal class label is the second instead of the last column')
generate.add_argument("--no-label", action='store_true', help='there is no label column present (data used only for predictions)')
generate.add_argument("--no-id", action='store_true', help='the first (second if label first) column is not an ID column')
# generate.add_argument("--tfidf", action='store_true', help="re-rank token counts using a regularized TF-IDF score")
# generate.add_argument("--cutoff", default=3, type=int, help="min. doc. frequency required to use a feature; "
#                                                             "value must be a positive integer; "
#                                                             "defaults to 3 (only use features seen at least in 3 documents)")
# generate.add_argument("--anova", action='store_true', help="use ANOVA F-values for feature weighting; default: chi^2")

# claset = parser.add_argument_group('LEARN/EVALUATE/PREDICT classifier options')
# claset.add_argument("--classifier", metavar='CLASS', choices=['ridge', 'svm', 'maxent', 'multinomial', 'bernoulli'], help="classifier to use (default: svm)")
# claset.add_argument("--feature-grid-search", action='store_true', help="run a grid search for the optimal feature parameters")
# claset.add_argument("--classifier-grid-search", action='store_true', help="run a grid search for the optimal classifier parameters")
# claset.add_argument("--model", metavar='FILE', type=str, help="save/load the classifier pipeline to/from disk")
learn.set_defaults(func=learn_model)

# evrep = parser.add_argument_group('EVALUATE evaluation and report options')
# evrep.add_argument("--parameters", action='store_true', help="report classification setup parameters")
# evrep.add_argument("--features", action='store_true', help="report feature size counts")
# evrep.add_argument("--top", metavar='N', default=0, type=int, help="list the N most significant features")
# evrep.add_argument("--worst", metavar='N', default=0, type=int, help="list the N least significant features")
# evrep.add_argument("--false-negatives", action='store_true', help="report false negative classifications")
# evrep.add_argument("--false-positives", action='store_true', help="report false positive classifications")
# evrep.add_argument("--classification-reports", action='store_true', help="print per-fold classification reports")
# evrep.add_argument("--folds", metavar='N', default=5, type=int, help="do N-fold cross-validation for internal evaluations; " "N must be an integer > 1; defaults to 5")
evaluate.set_defaults(func=evaluate_model)

predict.set_defaults(func=predict_labels)

# Argument Parsing
# ================

args = parser.parse_args()
args.func(args)

# if 2 > args.folds:
#     parser.error("the --folds value must be larger than one")
#
# if 1 > args.cutoff:
#     parser.error("the --cutoff value must be positive")
#
# if 1 > args.n_grams:
#     parser.error("the --n-grams value must be positive")
#
# if 1 > args.k_shingles:
#     parser.error("the --k-shingles value must be positive")
#
# if 0 > args.top:
#     parser.error("the --top parameter must be non-negative")
#
# if 0 > args.worst:
#     parser.error("the --worst parameter must be non-negative")
#
#
# patterns = None
#
# if args.patterns:
#     patterns = re.compile('|'.join(l.strip('\n\r') for l in args.patterns))
#     args.patterns.close()
#
# filehandles = []
#
# for path in args.groups:
#     try:
#         filehandles.append(open(path))
#     except Exception as e:
#         parser.error('failed to open "{}": {}'.format(path, e))
#
# data = Data(*filehandles,
#             columns=args.column, ngrams=args.n_grams,  # BIO-NER input
#             decap=args.decapitalize, patterns=patterns, mask=args.mask)  # plain-text input
#
# for fh in filehandles:
#     fh.close()
#
# classifier = None
# pipeline = []
# parameters = {}
# grid_search = args.feature_grid_search or args.classifier_grid_search
#
# # Classifier Setup
# # ================
#
# # TODO: defining all classifier parameters as opt-args for this script
#
# if args.classifier == 'ridge':
#     classifier = RidgeClassifier()
#
#     if args.classifier_grid_search:
#         parameters['classifier__alpha'] = [10., 1., 0., .1, .01]
#         parameters['classifier__normalize'] = [True, False]
#         parameters['classifier__solver'] = ['svd', 'cholesky', 'lsqr', 'sparse_cg']
#         parameters['classifier__tol'] = [.1, .01, 1e-3, 1e-6]
# elif args.classifier == 'svm':
#     classifier = LinearSVC(loss='l1')  # prefer Hinge loss (slower, but "better")
#
#     if args.classifier_grid_search:
#         parameters['classifier__C'] = [100., 10., 1., .1, .01]
#         parameters['classifier__class_weight'] = ['auto', None]
#         parameters['classifier__intercept_scaling'] = [10., 5., 1., .5]
#         parameters['classifier__penalty'] = ['l1', 'l2']
#         parameters['classifier__tol'] = [.1, .01, 1e-4, 1e-8]
# elif args.classifier == 'maxent':
#     classifier = LogisticRegression(class_weight='auto')
#
#     if args.classifier_grid_search:
#         parameters['classifier__C'] = [100., 10., 1., .1, .01]
#         parameters['classifier__class_weight'] = ['auto', None]
#         parameters['classifier__intercept_scaling'] = [10., 5., 1., .5]
#         parameters['classifier__penalty'] = ['l1', 'l2']
#         parameters['classifier__tol'] = [.1, .01, 1e-4, 1e-8]
# elif args.classifier == 'multinomial':
#     classifier = MultinomialNB()
#
#     if args.classifier_grid_search:
#         parameters['classifier__alpha'] = [10., 1., 0., .1, .01]
#         parameters['classifier__fit_prior'] = [True, False]
# elif args.classifier == 'bernoulli':
#     classifier = BernoulliNB()
#
#     if args.classifier_grid_search:
#         parameters['classifier__alpha'] = [10., 1., 0., .1, .01]
#         parameters['classifier__binarize'] = [True, False]
#         parameters['classifier__fit_prior'] = [True, False]
# elif os.path.isfile(args.classifier):
#
#     # Prediction [with an existing pipeline]
#     # ==========
#     Predict(data, joblib.load(args.classifier))
#     import sys
#     sys.exit(0)
#
# else:
#     parser.error("unrecognized classifier '%s'" % args.classifier)
#
# report = Report(args.parameters, args.top, args.worst,
#                 args.false_negatives, args.false_positives,
#                 args.classification_reports, args.folds)
#
# # Feature Extraction
# # ==================
#
# if args.column is None:
#     # token_pattern = r'\b\w[\w-]+\b' if not args.token_pattern else r'\S+'
#     stop_words = STOP_WORDS if args.stop_words else None
#     vec = CountVectorizer(binary=False,
#                           lowercase=not args.real_case,
#                           min_df=args.cutoff,
#                           ngram_range=(1, args.n_grams),
#                           stop_words=stop_words,
#                           strip_accents='unicode',
#                           token_pattern=args.token_pattern)
#
#     if args.feature_grid_search:
#         parameters['extract__binary'] = [True, False]
#         parameters['extract__lowercase'] = [True, False]
#         parameters['extract__min_df'] = [1, 2, 3, 5]
#         parameters['extract__ngram_range'] = [(1, 1), (1, 2), (1, 3)]
#         parameters['extract__stop_words'] = [None, STOP_WORDS]
# else:
#     vec = MinFreqDictVectorizer(min_freq=args.cutoff)
#
#     if args.feature_grid_search:
#         parameters['extract__min_freq'] = [1, 2, 3, 5]
#
#
# pipeline.append(('extract', vec))
#
# if report.parameters:
#     PrintParams(vec, report)
#
# if not grid_search:
#     data.extract(vec)
#
# # Feature Transformation
# # ======================
#
# if args.tfidf:
#     tfidf = TfidfTransformer(norm='l2',
#                              sublinear_tf=True,
#                              smooth_idf=True,
#                              use_idf=True)
#
#     if report.parameters:
#         print()
#         PrintParams(tfidf, report)
#
#     pipeline.append(('transform', tfidf))
#
#     if args.feature_grid_search:
#         parameters['transform__norm'] = [None, 'l1', 'l2']
#         parameters['transform__use_idf'] = [True, False]
#         parameters['transform__smooth_idf'] = [True, False]
#         parameters['transform__sublinear_tf'] = [True, False]
#
#     if not grid_search:
#         data.transform(tfidf)
#
# # Feature Selection
# # =================
#
# if args.max_fpr != 1.0:
#     # False-Positive Rate
#     fprs = SelectFpr(chi2 if not args.anova else f_classif,
#                      alpha=args.max_fpr)
#     pipeline.append(('select', fprs))
#     num_feats = 0 if grid_search else data.n_features
#
#     if report.parameters:
#         print()
#         PrintParams(fprs, report)
#
#     if not grid_search:
#         with warnings.catch_warnings():
#             # suppress the irrelevant duplicate p-values warning
#             warnings.simplefilter("ignore")
#             data.transform(fprs)
#
#         if args.features:
#             print('\npruned {}/{} features'.format(
#                 num_feats - data.n_features, num_feats
#             ))
#     elif args.feature_grid_search:
#         parameters['select__alpha'] = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
# elif args.num_features != 0:
#     # K-Best Features
#     kbest = SelectKBest(chi2 if not args.anova else f_classif,
#                         k=args.num_features)
#     pipeline.append(('select', kbest))
#
#     if report.parameters:
#         print()
#         PrintParams(kbest, report)
#
#     if not grid_search:
#         with warnings.catch_warnings():
#             # suppress the irrelevant duplicate p-values warning
#             warnings.simplefilter("ignore")
#             data.transform(kbest)
#     elif args.feature_grid_search:
#         parameters['select__k'] = [1e2, 1e3, 1e4, 1e5]
#
# if not grid_search and args.features:
#     print('\ngroup sizes:', ', '.join(map(str, data.sizes)))
#     print('extracted {} features from {} instances'.format(
#         data.n_features, data.n_instances
#     ))
#
# # Classification
# # ==============
#
# pipeline.append(('classifier', classifier))
# pipeline = Pipeline(pipeline)
#
# if report.parameters:
#     PrintParams(classifier, report)
#
# if grid_search:
#     print("\nGrid Search:")
#     print('\n'.join(
#         '{}: {}'.format(k, repr(v)) for k, v in parameters.items()
#     ))
#     GridSearch(data, pipeline, parameters, report)
# else:
#     Classify(data, classifier, report)
#
# if args.save:
#     if (report.top or report.worst):
#         print()
#         PrintFeatures(classifier, data, report)
#
#     joblib.dump(pipeline, args.save)