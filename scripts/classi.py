#!/usr/bin/env python3

"""A command-line tool for text classification."""

# License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)

import argparse
import logging
from classy import generate_data, learn_model, evaluate_model, predict_labels

__author__ = "Florian Leitner <florian.leitner@gmail.com>"
__verison__ = "1.0"

# Program Setup
# =============

parser = argparse.ArgumentParser(
    description="An AGPLv3 command-line tool for text classification."
)
parser.add_argument('--verbose', '-V', action='count', default=0,
                    help='increase log level [WARN]')
parser.add_argument('--quiet', '-Q', action='count', default=0,
                    help='decrease log level [WARN]')
parser.add_argument('--logfile', metavar='FILE',
                    help='log to file instead of <STDERR>')

commands = parser.add_subparsers(help='a command (use `CMD -h` to get help about a command):', metavar="CMD")

generate = commands.add_parser('generate', help='an inverted index from text input and/using a vocabulary', aliases=['g', 'ge', 'gen', 'gene', 'gener', 'genera', 'generat'])
generate.add_argument('data', metavar='FILE', nargs='*', help="input text file(s) (default: read from STDIN)")
generate.add_argument('-i', '--index', metavar='FILE', required=True, help="file to write the qgenerated inverted index to")
generate.add_argument("-v", "--vocabulary", metavar='FILE', type=str, help="vocabulary file (generated if absent, used if given)")
generate.add_argument("-r", "--replace", action='store_true', help="generate and replace vocabulary file, even if it exists")
generate.add_argument("--csv", action='store_true', help='input is Excel-like CSV with double-quoted escapes and string values in quotes (default: TSV without quotes)')
generate.add_argument("--title", action='store_true', help='input has a title row (default: no title row, use numbers instead)')
generate.add_argument("--decap", action='store_true', help="lowercase the first letter of each sentence")
generate.add_argument("--lowercase", action='store_true', help="lowercase all letters")
generate.add_argument("--encoding", type=str, default='utf-8', help="encoding of input files (default: UTF-8)")
generate.set_defaults(func=generate_data)
generate.add_argument("--n-grams", metavar='N', type=int, default=1, help="generate N-grams of all words in a sentence")
generate.add_argument("--k-shingles", metavar='K', type=int, default=1, help="generate all combinations of any K n-grams in the instance")
generate.add_argument("--annotate", metavar='COL', type=int, action='append', help="annotate text columns with the extra label from this/these column(s)")
generate.add_argument("--label-first", action='store_true', help='the nominal class label is the first instead of the last column')
generate.add_argument("--label-second", action='store_true', help='the nominal class label is the second instead of the last column')
generate.add_argument("--no-label", action='store_true', help='there is no label column present (data used only for predictions)')
generate.add_argument("--no-id", action='store_true', help='the first (second if label first) column is not an ID column')
generate.add_argument("--cutoff", default=3, type=int, help="min. doc. frequency required to use a feature; value must be a positive integer; defaults to 3 (only use features seen at least in 3 documents); only has an effect if a new vocabulary is (re-) generated")

learn = commands.add_parser('learn', help='a model from inverted index data', aliases=['l', 'le', 'lea', 'lear'])
learn.add_argument('index', metavar='INFILE', help="inverted index input file")
learn.add_argument('model', metavar='OUTFILE', help="model output file")
learn.add_argument("--tfidf", action='store_true', help="re-rank counts using a regularized TF-IDF score")
learn.add_argument("-v", "--vocabulary", metavar='FILE', type=str, help="vocabulary file (optional)")
learn.add_argument("-c", "--classifier", metavar='CLASS', default='svm', choices=['ridge', 'svm', 'maxent', 'multinomial', 'bernoulli'], help="classifier to use (default: svm)")
learn.add_argument("-g", "--grid-search", action='store_true', help="run a grid search for the optimal classifier parameters")
learn.set_defaults(func=learn_model)

evaluate = commands.add_parser('evaluate', help='by learing from an inverted index', aliases=['e', 'ev', 'eva', 'eval', 'evalu', 'evalua', 'evaluat'])
# evaluate.add_argument("--parameters", action='store_true', help="report classification setup parameters")
# evaluate.add_argument("--features", action='store_true', help="report feature size counts")
# evaluate.add_argument("--top", metavar='N', default=0, type=int, help="list the N most significant features")
# evaluate.add_argument("--worst", metavar='N', default=0, type=int, help="list the N least significant features")
# evaluate.add_argument("--false-negatives", action='store_true', help="report false negative classifications")
# evaluate.add_argument("--false-positives", action='store_true', help="report false positive classifications")
# evaluate.add_argument("--classification-reports", action='store_true', help="print per-fold classification reports")
# evaluate.add_argument("--folds", metavar='N', default=5, type=int, help="do N-fold cross-validation for internal evaluations; " "N must be an integer > 1; defaults to 5")
evaluate.set_defaults(func=evaluate_model)

predict = commands.add_parser('predict', help='unlabeled data using a learned model', aliases=['p', 'pr', 'pre', 'pred', 'predi', 'predic'])
predict.set_defaults(func=predict_labels)

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