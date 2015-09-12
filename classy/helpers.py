"""
.. py:module:: classy.helpers
   :synopsis: Helper functions to inspect vocabularies and indices.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

from .data import load_index, load_vocabulary
from sklearn.externals import joblib


def print_labels(args):
    print(', '.join(load_index(args.index).label_names))


def print_parameters(args):
    pipeline = joblib.load(args.model)

    for key, value in sorted(pipeline.get_params().items()):
        if '__' in key:
            print(key, repr(value), sep='=')


def print_vocabulary(args):
    for word in load_vocabulary(args.vocabulary):
        print(word)


def print_doc_ids(args):
    print('\n'.join(load_index(args.index).text_ids))
