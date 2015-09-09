from .data import load_index, load_vocabulary
from .generate import generate_data
from .select import select_features
from .learn import learn_model
from .evaluate import evaluate_model
from .predict import predict_labels
from .classifiers import CLASSIFIERS


def print_labels(args):
    print(', '.join(load_index(args.index).label_names))


def print_vocabulary(args):
    for word in load_vocabulary(args.vocabulary):
        print(word)


def print_doc_ids(args):
    print('\n'.join(load_index(args.index).text_ids))
