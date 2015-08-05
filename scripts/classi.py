#!/usr/bin/env python3

"""A command-line tool for text classification."""

import argparse
import os.path
import re
import warnings

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFpr, SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

from classy import \
    Classify, Data, GridSearch, Predict, \
    PrintParams, Report, STOP_WORDS, PrintFeatures, MinFreqDictVectorizer

__author__ = "Florian Leitner <florian.leitner@gmail.com>"
__verison__ = "1.0"

# Program Setup
# =============

parser = argparse.ArgumentParser(
    description=__file__.doc,
    usage="%(prog)s [OPTIONS] CLASS [FILE...]"
)

parser.add_argument("classifier", metavar='CLASS', help="choices: ridge, svm, maxent, multinomial, bernoulli, or a FILE from which to load a saved model")
parser.add_argument("data", metavar='FILE', nargs='*', help="file(s) containing data (read from STDIN if absent)")

featgen = parser.add_argument_group('options for the feature generation from the input')
featgen.add_argument("--csv", action='store_true', help='read Excel-like CSV with double-quote escapes and strings in quotes (default: TSV without quotes)')
featgen.add_argument("--class-first", action='store_true', help='the nominal class label is the first instead of the last column')
featgen.add_argument("--decapitalize", action='store_true', help="lowercase the first letter of each sentence")
featgen.add_argument("--lowercase", action='store_true', help="lowercase all letters")
featgen.add_argument("--n-grams", metavar='N', type=int, default=1, help="generate N-grams of all words in a sentence")
featgen.add_argument("--k-shingles", metavar='K', type=int, default=1, help="generate all combinations of any K n-grams in the instance")
featgen.add_argument("--tfidf", action='store_true', help="re-rank token counts using a regularized TF-IDF score")
featgen.add_argument("--cutoff", default=3, type=int, help="min. doc. frequency required to use a feature; " "value must be a positive integer; " "defaults to 3 (only use features seen at least in 3 documents)")
featgen.add_argument("--anova", action='store_true', help="use ANOVA F-values for feature weighting; default: chi^2")

claset = parser.add_argument_group('general classifier options')
claset.add_argument("--feature-grid-search", action='store_true', help="run a grid search for the optimal feature parameters")
claset.add_argument("--classifier-grid-search", action='store_true', help="run a grid search for the optimal classifier parameters")
claset.add_argument("--save", metavar='FILE', type=str, help="store the resulting classifier pipeline on disk")

evrep = parser.add_argument_group('evaluation and report')
evrep.add_argument("--parameters", action='store_true', help="report classification setup parameters")
evrep.add_argument("--features", action='store_true', help="report feature size counts")
evrep.add_argument("--top", metavar='N', default=0, type=int, help="list the N most significant features")
evrep.add_argument("--worst", metavar='N', default=0, type=int, help="list the N least significant features")
evrep.add_argument("--false-negatives", action='store_true', help="report false negative classifications")
evrep.add_argument("--false-positives", action='store_true', help="report false positive classifications")
evrep.add_argument("--classification-reports", action='store_true', help="print per-fold classification reports")
evrep.add_argument("--folds", metavar='N', default=5, type=int, help="do N-fold cross-validation for internal evaluations; " "N must be an integer > 1; defaults to 5")

# Argument Parsing
# ================

args = parser.parse_args()

if 2 > args.folds:
    parser.error("the --folds value must be larger than one")

if 1 > args.cutoff:
    parser.error("the --cutoff value must be positive")

if 1 > args.n_grams:
    parser.error("the --n-grams value must be positive")

if 1 > args.k_shingles:
    parser.error("the --k-shingles value must be positive")

if 0 > args.top:
    parser.error("the --top parameter must be non-negative")

if 0 > args.worst:
    parser.error("the --worst parameter must be non-negative")


patterns = None

if args.patterns:
    patterns = re.compile('|'.join(l.strip('\n\r') for l in args.patterns))
    args.patterns.close()

filehandles = []

for path in args.groups:
    try:
        filehandles.append(open(path))
    except Exception as e:
        parser.error('failed to open "{}": {}'.format(path, e))

data = Data(*filehandles,
            columns=args.column, ngrams=args.n_grams,  # BIO-NER input
            decap=args.decapitalize, patterns=patterns, mask=args.mask)  # plain-text input

for fh in filehandles:
    fh.close()

classifier = None
pipeline = []
parameters = {}
grid_search = args.feature_grid_search or args.classifier_grid_search

# Classifier Setup
# ================

# TODO: defining all classifier parameters as opt-args for this script

if args.classifier == 'ridge':
    classifier = RidgeClassifier()

    if args.classifier_grid_search:
        parameters['classifier__alpha'] = [10., 1., 0., .1, .01]
        parameters['classifier__normalize'] = [True, False]
        parameters['classifier__solver'] = ['svd', 'cholesky', 'lsqr', 'sparse_cg']
        parameters['classifier__tol'] = [.1, .01, 1e-3, 1e-6]
elif args.classifier == 'svm':
    classifier = LinearSVC(loss='l1')  # prefer Hinge loss (slower, but "better")

    if args.classifier_grid_search:
        parameters['classifier__C'] = [100., 10., 1., .1, .01]
        parameters['classifier__class_weight'] = ['auto', None]
        parameters['classifier__intercept_scaling'] = [10., 5., 1., .5]
        parameters['classifier__penalty'] = ['l1', 'l2']
        parameters['classifier__tol'] = [.1, .01, 1e-4, 1e-8]
elif args.classifier == 'maxent':
    classifier = LogisticRegression(class_weight='auto')

    if args.classifier_grid_search:
        parameters['classifier__C'] = [100., 10., 1., .1, .01]
        parameters['classifier__class_weight'] = ['auto', None]
        parameters['classifier__intercept_scaling'] = [10., 5., 1., .5]
        parameters['classifier__penalty'] = ['l1', 'l2']
        parameters['classifier__tol'] = [.1, .01, 1e-4, 1e-8]
elif args.classifier == 'multinomial':
    classifier = MultinomialNB()

    if args.classifier_grid_search:
        parameters['classifier__alpha'] = [10., 1., 0., .1, .01]
        parameters['classifier__fit_prior'] = [True, False]
elif args.classifier == 'bernoulli':
    classifier = BernoulliNB()

    if args.classifier_grid_search:
        parameters['classifier__alpha'] = [10., 1., 0., .1, .01]
        parameters['classifier__binarize'] = [True, False]
        parameters['classifier__fit_prior'] = [True, False]
elif os.path.isfile(args.classifier):

    # Prediction [with an existing pipeline]
    # ==========
    Predict(data, joblib.load(args.classifier))
    import sys
    sys.exit(0)

else:
    parser.error("unrecognized classifier '%s'" % args.classifier)

report = Report(args.parameters, args.top, args.worst,
                args.false_negatives, args.false_positives,
                args.classification_reports, args.folds)

# Feature Extraction
# ==================

if args.column is None:
    # token_pattern = r'\b\w[\w-]+\b' if not args.token_pattern else r'\S+'
    stop_words = STOP_WORDS if args.stop_words else None
    vec = CountVectorizer(binary=False,
                          lowercase=not args.real_case,
                          min_df=args.cutoff,
                          ngram_range=(1, args.n_grams),
                          stop_words=stop_words,
                          strip_accents='unicode',
                          token_pattern=args.token_pattern)

    if args.feature_grid_search:
        parameters['extract__binary'] = [True, False]
        parameters['extract__lowercase'] = [True, False]
        parameters['extract__min_df'] = [1, 2, 3, 5]
        parameters['extract__ngram_range'] = [(1, 1), (1, 2), (1, 3)]
        parameters['extract__stop_words'] = [None, STOP_WORDS]
else:
    vec = MinFreqDictVectorizer(min_freq=args.cutoff)

    if args.feature_grid_search:
        parameters['extract__min_freq'] = [1, 2, 3, 5]


pipeline.append(('extract', vec))

if report.parameters:
    PrintParams(vec, report)

if not grid_search:
    data.extract(vec)

# Feature Transformation
# ======================

if args.tfidf:
    tfidf = TfidfTransformer(norm='l2',
                             sublinear_tf=True,
                             smooth_idf=True,
                             use_idf=True)

    if report.parameters:
        print()
        PrintParams(tfidf, report)

    pipeline.append(('transform', tfidf))

    if args.feature_grid_search:
        parameters['transform__norm'] = [None, 'l1', 'l2']
        parameters['transform__use_idf'] = [True, False]
        parameters['transform__smooth_idf'] = [True, False]
        parameters['transform__sublinear_tf'] = [True, False]

    if not grid_search:
        data.transform(tfidf)

# Feature Selection
# =================

if args.max_fpr != 1.0:
    # False-Positive Rate
    fprs = SelectFpr(chi2 if not args.anova else f_classif,
                     alpha=args.max_fpr)
    pipeline.append(('select', fprs))
    num_feats = 0 if grid_search else data.n_features

    if report.parameters:
        print()
        PrintParams(fprs, report)

    if not grid_search:
        with warnings.catch_warnings():
            # suppress the irrelevant duplicate p-values warning
            warnings.simplefilter("ignore")
            data.transform(fprs)

        if args.features:
            print('\npruned {}/{} features'.format(
                num_feats - data.n_features, num_feats
            ))
    elif args.feature_grid_search:
        parameters['select__alpha'] = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
elif args.num_features != 0:
    # K-Best Features
    kbest = SelectKBest(chi2 if not args.anova else f_classif,
                        k=args.num_features)
    pipeline.append(('select', kbest))

    if report.parameters:
        print()
        PrintParams(kbest, report)

    if not grid_search:
        with warnings.catch_warnings():
            # suppress the irrelevant duplicate p-values warning
            warnings.simplefilter("ignore")
            data.transform(kbest)
    elif args.feature_grid_search:
        parameters['select__k'] = [1e2, 1e3, 1e4, 1e5]

if not grid_search and args.features:
    print('\ngroup sizes:', ', '.join(map(str, data.sizes)))
    print('extracted {} features from {} instances'.format(
        data.n_features, data.n_instances
    ))

# Classification
# ==============

pipeline.append(('classifier', classifier))
pipeline = Pipeline(pipeline)

if report.parameters:
    PrintParams(classifier, report)

if grid_search:
    print("\nGrid Search:")
    print('\n'.join(
        '{}: {}'.format(k, repr(v)) for k, v in parameters.items()
    ))
    GridSearch(data, pipeline, parameters, report)
else:
    Classify(data, classifier, report)

if args.save:
    if (report.top or report.worst):
        print()
        PrintFeatures(classifier, data, report)

    joblib.dump(pipeline, args.save)