#!/usr/bin/env python3

"""A command-line tool for text classification."""

# License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)

import argparse
import logging
from classy import CLASSIFIERS, generate_data, learn_model, evaluate_model, predict_labels
from classy.data import load_index

__author__ = "Florian Leitner <florian.leitner@gmail.com>"
__verison__ = "1.0"

# Program Setup
# =============


parser = argparse.ArgumentParser(
    description="An AGPLv3 command-line tool for text classification."
)
parser.add_argument('--verbose', '-V', action='count', default=0, help='increase log level [WARN]')
parser.add_argument('--quiet', '-Q', action='count', default=0, help='decrease log level [WARN]')
parser.add_argument('--logfile', metavar='FILE', help='log to file instead of <STDERR>')
parser.add_argument("--vocabulary", '-v', metavar='VOCAB', type=str, help="vocabulary file (to generate or use)")
#parser.set_defaults(func=lambda _: parser.print_help())

commands = parser.add_subparsers(help='a command (use `CMD -h` to get help about a command):', metavar="CMD")

generate = commands.add_parser('generate', help='an inverted index from text input and/using a vocabulary', aliases=['g', 'ge', 'gen', 'gene', 'gener', 'genera', 'generat'])
generate.add_argument('index', metavar='INDEX', help="file (names) to write the generated inverted index to")
generate.add_argument('data', metavar='TEXT', nargs='*', help="input text file(s) (default: read from STDIN)")
generate.add_argument("-r", "--replace", action='store_true', help="replace/regenerate the vocabulary file if it exists")
generate.add_argument("--csv", action='store_true', help='input is Excel-like CSV with double-quoted string values, using two double-quotes to insert quotes inside the text (default: tab-separated, without quoted strings. using escaped tabs to add tabs inside strings)')
generate.add_argument("--title", action='store_true', help='input has a title row (default: no title row, use numbers instead)')
generate.add_argument("--decap", action='store_true', help="lowercase the first letter of each sentence")
generate.add_argument("--lowercase", action='store_true', help="lowercase all letters")
generate.add_argument("--encoding", type=str, default='utf-8', help="encoding of input files (default: %(default)s)")
generate.add_argument("--n-grams", metavar='N', type=int, default=2, help="generate N-grams of all words in a sentence (default=%(default)s)")
generate.add_argument("--k-shingles", metavar='K', type=int, default=1, help="generate all combinations of any K n-grams in the instance (default=%(default)s)")
generate.add_argument("--annotate", metavar='COL', type=int, action='append', help="annotate text columns with the extra label from this/these column(s) using 1-based count")
generate.add_argument("--label-first", action='store_true', help='the nominal class label is the first instead of the last column')
generate.add_argument("--label-second", action='store_true', help='the nominal class label is the second instead of the last column')
generate.add_argument("--no-label", action='store_true', help='there is no label column present (data used only for predictions)')
generate.add_argument("--no-id", action='store_true', help='the first (second if label first) column is not an ID column')
generate.add_argument("--cutoff", default=3, type=int, help="min. doc. frequency required to use a feature; value must be a positive integer;  (default=%(default)s - only use features seen at least in 3 documents); only has an effect if a new vocabulary is (re-) generated")
generate.set_defaults(func=generate_data)

learn = commands.add_parser('learn', help='a model from inverted index data', aliases=['l', 'le', 'lea', 'lear'])
learn.add_argument('index', metavar='INDEX', help="inverted index input file")
learn.add_argument('model', metavar='MODEL', help="model output file")
learn.add_argument("--tfidf", action='store_true', help="re-rank counts using a regularized TF-IDF score")
learn.add_argument("--classifier", default='svm', choices=CLASSIFIERS.keys(), help="classifier to use (default=%(default)s)")
learn.add_argument("-g", "--grid-search", action='store_true', help="run a 5xCV grid search to fit the optimal parameters before storing the model")
learn.set_defaults(func=learn_model)

evaluate = commands.add_parser('evaluate', help='a trained model on unseen data or a classifier via CV (no model)', aliases=['e', 'ev', 'eva', 'eval', 'evalu', 'evalua', 'evaluat'])
evaluate.add_argument('index', metavar='INDEX', help="inverted index input file")
evaluate.add_argument('model', metavar='MODEL', nargs='?', help="trained model file; if absent: cross-validation (CV)")
evaluate.add_argument("--pr-curve", action='store_true', help="plot the PR curve (requires a working and configured matplotlib installation; no effect during CV)")
cross_validation = evaluate.add_argument_group('optional arguments for cross-validation (CV; when no model file is given)', None)
cross_validation.add_argument("--classifier", default='svm', choices=CLASSIFIERS.keys(), help="classifier to use (default: %(default)s)")
cross_validation.add_argument("--folds", metavar='N', default=5, type=int, help="CV folds; N must be an integer > 1; (default=%(default)s)")
cross_validation.add_argument("--tfidf", action='store_true', help="re-rank counts using a regularized TF-IDF score")

evaluate.set_defaults(func=evaluate_model)

predict = commands.add_parser('predict', help='unlabeled data using a learned model (printing text ID - tab - label lines)', aliases=['p', 'pr', 'pre', 'pred', 'predi', 'predic'])
predict.add_argument('model', metavar='MODEL', help="trained model file")
predict.add_argument('index', metavar='IDX/TXT', nargs='*', help="one inverted index input file, one or more text file(s); if absent: read from STDIN (--text mode only)")
predict.add_argument("--text", action='store_true', help='IDX/TXT is a/are text file(s) or STDIN (instead of an inverted index); runs the predictor in streaming mode')
predict.add_argument("--scores", action='store_true', help="add the classifier's score for each label as column(s; multinomial classification) in the output")
predict_text = predict.add_argument_group('optional arguments for --text', "It is recommendable to use the same options as for generating the inverted index used to learn the MODEL.")
predict_text.add_argument("--label", metavar='NAME', type=str, action='append', help="name for n-th label: Internally, classi.py uses 0-based integers per class; the label names used in the labeled training data are listed in the order of those integers to STDOUT while generating the index; if the option is not used, the internal integers are output as the text labels. Existing labels in a labeled inverted index can also be listed with the command `labels`.")
predict_text.add_argument("--csv", action='store_true', help='input is Excel-like CSV with double-quoted string values, using two double-quotes to insert quotes inside the text (default: tab-separated, without quoted strings. using escaped tabs to add tabs inside strings)')
predict_text.add_argument("--title", action='store_true', help='input has a title row (default: no title row, use numbers instead)')
predict_text.add_argument("--decap", action='store_true', help="lowercase the first letter of each sentence")
predict_text.add_argument("--lowercase", action='store_true', help="lowercase all letters")
predict_text.add_argument("--encoding", type=str, default='utf-8', help="encoding of input files (default: %(default)s)")
predict_text.add_argument("--n-grams", metavar='N', type=int, default=2, help="generate N-grams of all words in a sentence (default=%(default)s)")
predict_text.add_argument("--k-shingles", metavar='K', type=int, default=1, help="generate all combinations of any K n-grams in the instance (default=%(default)s)")
predict_text.add_argument("--annotate", metavar='COL', type=int, action='append', help="annotate text columns with the extra label from this/these column(s) using 1-based count")
predict_text.add_argument("--id-second", action='store_true', help='the text ID is the second instead of the first column')
predict_text.add_argument("--id-last", action='store_true', help='the text ID is the last instead of the first column')
predict_text.add_argument("--no-id", action='store_true', help='the input has no ID column')
predict.set_defaults(func=predict_labels)

predict = commands.add_parser('labels', help='list all label names (classes) in a (labeled) index file')
predict.add_argument('index', metavar='INDEX', help="inverted index file")
predict.set_defaults(func=lambda a: print(', '.join(load_index(a.index).label_names)))


# Argument Parsing
# ================

args = parser.parse_args()

log_format = '%(levelname)-8s %(module) 10s: %(funcName)s %(message)s'
log_adjust = max(min(args.quiet - args.verbose, 2), -2) * 10
logging.basicConfig(filename=args.logfile, level=logging.WARNING + log_adjust,
                    format=log_format)
logging.info('verbosity increased')
logging.debug('verbosity increased')

args.func(args)