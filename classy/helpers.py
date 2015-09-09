"""
.. py:module:: classy.helpers
   :synopsis: Trivial helper functions.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

from .data import load_index, load_vocabulary


def print_labels(args):
    print(', '.join(load_index(args.index).label_names))


def print_vocabulary(args):
    for word in load_vocabulary(args.vocabulary):
        print(word)


def print_doc_ids(args):
    print('\n'.join(load_index(args.index).text_ids))
