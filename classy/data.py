"""
.. py:module:: classy.data
   :synopsis: Data I/O for the text classifiers.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
import pickle
from numpy.core.umath import isfinite
from numpy import typecodes, asarray

L = logging.getLogger(__name__)


def check_shapes(doc_ids, index, labels):
    L.info("index shape: %s doc_ids: %s, labels: %s", index.shape,
           'None' if doc_ids is None else len(doc_ids),
           'None' if labels is None else len(labels))

    if doc_ids is not None and labels is not None:
        if len(doc_ids) != len(labels):
            msg = 'length of IDs (%d) != length of labels (%d)'
            raise ValueError(msg % (len(doc_ids), len(labels)))

    if doc_ids is not None:
        if len(doc_ids) != index.shape[0]:
            msg = 'length of IDs (%d) != number of index rows (%d)'
            raise ValueError(msg % (len(doc_ids), index.shape[0]))

    if labels is not None:
        if len(labels) != index.shape[0]:
            msg = 'length of labels (%d) != number of index rows (%d)'
            raise ValueError(msg % (len(labels), len(index.shape[0])))

    if index.dtype.char in typecodes['AllFloat'] and \
            not isfinite(index.sum()) and \
            not isfinite(index).all():
        raise ValueError("index contains NaN, infinity"
                         " or a value too large for %r." % index.dtype)


def save_index(doc_ids, labels, index, path):
    check_shapes(doc_ids, index, labels)
    L.info("saving inverted index to '%s'", path)

    with open(path, 'wb') as f:
        pickle.dump([doc_ids, asarray(labels, str), index], f)


def load_index(path):
    L.info("loading inverted index from '%s'", path)

    with open(path, 'rb') as f:
        doc_ids, labels, index = pickle.load(f)

    check_shapes(doc_ids, index, labels)
    return doc_ids, labels, index


def save_vocabulary(vocabulary, index, path):
    L.info(" vocabulary size: %s", len(vocabulary))
    L.debug("vocabulary: %s", vocabulary.keys())
    L.info("saving vocabulary to '%s'", path)

    if len(vocabulary) != index.shape[1]:
        msg = 'length of vocabulary (%d) != number of index columns (%d)'
        raise ValueError(msg % (len(vocabulary), index.shape[1]))

    with open(path, 'wb') as f:
        pickle.dump(vocabulary, f)


def load_vocabulary(path, index):
    L.info("loading vocabulary from '%s'", path)

    with open(path, 'rb') as f:
        vocabulary = pickle.load(f)

    L.info(" vocabulary size: %s", len(vocabulary))

    if len(vocabulary) != index.shape[1]:
        msg = 'length of vocabulary (%d) != number of index columns (%d)'
        raise ValueError(msg % (len(vocabulary), index.shape[1]))

    return vocabulary
