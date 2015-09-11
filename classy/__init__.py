"""The swiss-knife of text classification."""

# License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)

import argparse
import logging
from classy.classifiers import CLASSIFIERS
from classy.evaluate import evaluate_model
from classy.generate import generate_data
from classy.helpers import print_labels, print_parameters, print_vocabulary, print_doc_ids
from classy.learn import learn_model
from classy.predict import predict_labels
from classy.select import select_features


__author__ = "Florian Leitner <florian.leitner@gmail.com>"
__version__ = "1.0"

def main():
    parser = argparse.ArgumentParser(description="An AGPLv3 command-line tool for text classification.")
    parser.add_argument('--verbose', '-V', action='count', default=0, help='increase log level [WARN]')
    parser.add_argument('--quiet', '-Q', action='count', default=0, help='decrease log level [WARN]')
    parser.add_argument('--logfile', metavar='FILE', help='log to file instead of <STDERR>')
    parser.add_argument("--jobs", '-j', metavar='N', type=int, default=-1, help="number of threads/jobs to use for parallel processes (default: as many as [hyper-]cores)")

    commands = parser.add_subparsers(help='a command (use `CMD -h` to get help about a command):', metavar="CMD")

    generate = commands.add_parser('generate', help='an inverted index from text input and/using a vocabulary', aliases=['g', 'ge', 'gen', 'gene', 'gener', 'genera', 'generat'])
    generate.add_argument('index', metavar='INDEX', help="file (names) to write the generated inverted index to")
    generate.add_argument('data', metavar='TEXT', nargs='*', help="input text file(s) (default: read from STDIN)")
    generate.add_argument("--vocabulary", '-v', metavar='VOCAB', type=str, help="vocabulary file to generate (with --replace or if the file does not exist) or use (e.g., to generate a test set index with this input vocabulary that was used to create the training set; For those cases, there is no need to --select or --cutoff)")
    generate.add_argument("--replace", '-r', action='store_true', help="replace/regenerate the vocabulary file if it exists")
    generate.add_argument("--csv", action='store_true', help='input is Excel-like CSV with double-quoted string values, using two double-quotes to insert quotes inside the text (default: tab-separated, without quoted strings. using escaped tabs to add tabs inside strings)')
    generate.add_argument("--title", action='store_true', help='input has a title row (default: no title row, use column numbers as column names instead)')
    generate.add_argument("--decap", action='store_true', help="lowercase the first letter of each sentence")
    generate.add_argument("--lowercase", action='store_true', help="lowercase all letters")
    generate.add_argument("--encoding", type=str, default='utf-8', help="encoding of input files (default: %(default)s)")
    generate.add_argument("--n-grams", '-n', metavar='N', type=int, default=1, help="generate N-grams of all words in a sentence (default=%(default)s)")
    generate.add_argument("--k-shingles", '-k', metavar='K', type=int, default=1, help="generate all combinations of any K n-grams in the instance (default=%(default)s)")
    generate.add_argument("--annotate", '-a', metavar='COL', type=int, action='append', help="append the string from annotation column(s) (1-based column count) to each text-based feature generated")
    generate.add_argument("--feature", '-f', metavar='COL', type=int, action='append', help="generate independent vocabulary features from annotation column(s) (1-based column count)")
    generate.add_argument("--binarize", action='store_true', help='combine all feature annotations to paired features')
    generate.add_argument("--label-first", action='store_true', help='the nominal class label is the first instead of the last column')
    generate.add_argument("--label-second", action='store_true', help='the nominal class label is the second instead of the last column')
    generate.add_argument("--no-label", action='store_true', help='there is no label column present (i.e., the index is used only for predictions)')
    generate.add_argument("--no-id", action='store_true', help='there is no ID column (use sequential numbering as text IDs)')
    generate.add_argument("--cutoff", '-c', default=1, metavar='C', type=int, help="drop features below min. document frequency C; value must be a positive integer (default: keep all features)")
    generate.add_argument("--select", '-s', default=0, metavar='S', type=int, help="select the S best features using a chi-square test; value must be a positive integer (default: keep all features)")
    generate.add_argument("--eliminate", '-e', default=0, metavar='E', type=int, help="recursively eliminate down to E features (default: keep all features)")
    generate.set_defaults(func=generate_data)

    select = commands.add_parser('select', help='feature selection to reduce the index and vocabulary size', aliases=['s', 'se', 'sel', 'sele', 'selec'])
    select.add_argument('index', metavar='IN_INDEX', help="the inverted index file to read and prune")
    select.add_argument('new_index', metavar='OUT_INDEX', help="the reduced inverted index file to write")
    select.add_argument("vocabulary", metavar='IN_VOCAB', nargs='?', help="vocabulary file of the input index (optional, to create a reduced vocabulary)")
    select.add_argument('new_vocabulary', metavar='OUT_VOCAB', nargs='?', help="the reduced vocabulary to write (if the original one is given)")
    select.add_argument("--cutoff", '-c', default=1, metavar='C', type=int, help="drop features below min. document frequency C; value must be a positive integer (default: keep all features)")
    select.add_argument("--select", '-s', default=0, metavar='S', type=int, help="select the S best features using a chi-square test; value must be a positive integer (default: keep all features)")
    select.add_argument("--eliminate", '-e', default=0, metavar='E', type=int, help="recursively eliminate down to E features (default: keep all features)")
    select.set_defaults(func=select_features)

    learn = commands.add_parser('learn', help='a model from inverted index data', aliases=['l', 'le', 'lea', 'lear'])
    learn.add_argument('index', metavar='INDEX', help="inverted index input file")
    learn.add_argument('model', metavar='MODEL', help="model output file")
    learn.add_argument("--vocabulary", '-v', metavar='VOCAB', type=str, help="vocabulary file to use for (optional) feature reports")
    learn.add_argument("--tfidf", action='store_true', help="re-rank counts using a regularized TF-IDF score")
    learn.add_argument("--scale", action='store_true', help="scale the features to norm")
    learn.add_argument("--filter", action='store_true', help="filter features using a L1-based, linear estimator (linear SVM or Logistic Regression)")
    learn.add_argument("--classifier", '-c', default='svm', choices=CLASSIFIERS.keys(), help="classifier to use (default=%(default)s)")
    learn.add_argument("--parameters", '-p', type=str, help="comma-separated parameter string (e.g. \"classify__loss='hinge',scale__norm='l2',transform__sublinear_tf=True\")")
    learn.add_argument("--grid-search", '-g', action='store_true', help="run a 5xCV grid search to fit the optimal parameters before storing the model")
    learn.set_defaults(func=learn_model)

    evaluate = commands.add_parser('evaluate', help='a trained model on unseen data or a classifier via CV (no model)', aliases=['e', 'ev', 'eva', 'eval', 'evalu', 'evalua', 'evaluat'])
    evaluate.add_argument('index', metavar='INDEX', help="inverted index input file")
    evaluate.add_argument('model', metavar='MODEL', nargs='?', help="trained model file; if absent: cross-validation (CV)")
    evaluate.add_argument("--pr-curve", '-p', action='store_true', help="plot the PR curve (requires a working and configured matplotlib installation; no effect during CV)")
    cross_validation = evaluate.add_argument_group('optional arguments for cross-validation (CV; when no model file is given)', None)
    cross_validation.add_argument("--vocabulary", '-v', metavar='VOCAB', type=str, help="vocabulary file to use for (optional) cross-validation feature reports")
    cross_validation.add_argument("--classifier", '-c', default='svm', choices=CLASSIFIERS.keys(), help="classifier to use (default: %(default)s)")
    cross_validation.add_argument("--folds", metavar='N', default=5, type=int, help="CV folds; N must be an integer > 1; (default=%(default)s)")
    cross_validation.add_argument("--tfidf", action='store_true', help="re-rank counts using a regularized TF-IDF score")
    cross_validation.add_argument("--scale", action='store_true', help="scale the features to norm")
    cross_validation.add_argument("--filter", action='store_true', help="filter features using a L1-based, linear estimator (linear SVM or Logistic Regression)")
    evaluate.set_defaults(func=evaluate_model)

    predict = commands.add_parser('predict', help='unlabeled data using a learned model (printing text ID - tab - label lines)', aliases=['p', 'pr', 'pre', 'pred', 'predi', 'predic'])
    predict.add_argument('model', metavar='MODEL', help="trained model file")
    predict.add_argument('index', metavar='IDX/TXT', nargs='*', help="one inverted index input file, one or more text file(s); if absent: read from STDIN (--text mode only)")
    predict.add_argument("--vocabulary", '-v', metavar='VOCAB', type=str, help="vocabulary file to use for stream-based prediction from text files")
    predict.add_argument("--text", '-t', action='store_true', help='IDX/TXT is a/are text file(s) or STDIN (instead of an inverted index); runs the predictor in streaming mode')
    predict.add_argument("--scores", '-s', action='store_true', help="add the classifier's score for each label as column(s; multinomial classification) in the output")
    predict_text = predict.add_argument_group('optional arguments for --text', "It is recommendable to use the same options as for generating the inverted index used to learn the MODEL.")
    predict_text.add_argument("--label", '-l', metavar='NAME', type=str, action='append', help="name for n-th label: Internally, classi.py uses 0-based integers per class; the label names used in the labeled training data are listed in the order of those integers to STDOUT while generating the index; if the option is not used, the internal integers are output as the text labels. Existing labels in a labeled inverted index can also be listed with the command `labels`.")
    predict_text.add_argument("--csv", action='store_true', help='input is Excel-like CSV with double-quoted string values, using two double-quotes to insert quotes inside the text (default: tab-separated, without quoted strings. using escaped tabs to add tabs inside strings)')
    predict_text.add_argument("--title", action='store_true', help='input has a title row (default: no title row, use numbers instead)')
    predict_text.add_argument("--decap", action='store_true', help="lowercase the first letter of each sentence")
    predict_text.add_argument("--lowercase", action='store_true', help="lowercase all letters")
    predict_text.add_argument("--encoding", '-e', type=str, default='utf-8', help="encoding of input files (default: %(default)s)")
    predict_text.add_argument("--n-grams", '-n', metavar='N', type=int, default=2, help="generate N-grams of all words in a sentence (default=%(default)s)")
    predict_text.add_argument("--k-shingles", '-k', metavar='K', type=int, default=1, help="generate all combinations of any K n-grams in the instance (default=%(default)s)")
    predict_text.add_argument("--annotate", '-a', metavar='COL', type=int, action='append', help="annotate text columns with the extra label from this/these column(s) using 1-based count")
    predict_text.add_argument("--feature", '-f', metavar='COL', type=int, action='append', help="generate independent vocabulary features from annotation column(s) (1-based column count)")
    predict_text.add_argument("--binarize", action='store_true', help='combine all feature annotations to paired features')
    predict_text.add_argument("--id-second", action='store_true', help='the text ID is the second instead of the first column')
    predict_text.add_argument("--id-last", action='store_true', help='the text ID is the last instead of the first column')
    predict_text.add_argument("--no-id", action='store_true', help='the input has no ID column')
    predict.set_defaults(func=predict_labels)

    labels = commands.add_parser('labels', help='list all label names (classes) in a (labeled) index file')
    labels.add_argument('index', metavar='INDEX', help="inverted index file")
    labels.set_defaults(func=print_labels)

    parameters = commands.add_parser('parameters', help='list all parameters of a model file', aliases=['params'])
    parameters.add_argument('model', metavar='MODEL', help="model file")
    parameters.set_defaults(func=print_parameters)

    vocabulary = commands.add_parser('vocabulary', help='list all vocabulary words (features)', aliases=['v', 'vo', 'voc', 'voca', 'vocab'])
    vocabulary.add_argument('vocabulary', metavar='VOCAB', help="vocabulary file")
    vocabulary.set_defaults(func=print_vocabulary)

    documents = commands.add_parser('documents', help='list all document IDs', aliases=['d', 'do', 'doc', 'docs', 'ids'])
    documents.add_argument('index', metavar='INDEX', help="inverted index file")
    documents.set_defaults(func=print_doc_ids)

    args = parser.parse_args()

    log_format = "%(asctime)s %(levelname)s %(module)s.%(funcName)s: %(message)s"
    formatter = logging.Formatter(log_format, "%y-%m-%d %H:%M:%S")
    log_adjust = max(min(args.quiet - args.verbose, 2), -2) * 10
    logging.basicConfig(filename=args.logfile, level=logging.WARNING + log_adjust,
                        format=log_format)
    logging.getLogger('').handlers[0].setFormatter(formatter)
    logging.info('verbosity increased')
    logging.debug('verbosity increased')

    try:
        args.func(args)
    except AttributeError:
        parser.print_usage()
