"""
.. py:module:: classy.data
   :synopsis: Data I/O for the text classifiers.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""
from collections import namedtuple, Counter

import logging
import pickle
from numpy.core.umath import isfinite
from numpy import typecodes, asarray, unique, where
from sklearn.preprocessing import label_binarize

L = logging.getLogger(__name__)
Data = namedtuple('Data', 'text_ids index labels label_names min_label')


def make_data(inverted_index, text_ids=None, labels=None):
    if labels is not None:
        labels = asarray(labels, str)
        # noinspection PyTupleAssignmentBalance
        uniq, inverse = unique(labels, return_inverse=True)
        n_classes = len(uniq)
        min_label = Counter(labels).most_common()[-1][0]
        min_label = where(uniq == min_label)[0][0]

        if n_classes == 1:
            L.warn("only one target label", uniq[0])
        elif n_classes > 2:
            labels = label_binarize(labels, classes=uniq)
        else:
            labels = inverse

        print('ordered labels:', ', '.join(uniq))
    else:
        uniq, min_label = None, None

    return Data(text_ids, inverted_index.tocsr(), labels, uniq, min_label)


def get_n_rows(data):
    return data.index.shape[0]


def get_n_cols(data):
    return data.index.shape[1]


def check_integrity(data):
    L.info("index shape: %s doc_ids: %s, labels: %s", data.index.shape,
           'None' if data.text_ids is None else len(data.text_ids),
           'None' if data.labels is None else len(data.labels))
    rows = get_n_rows(data)

    if data.text_ids is not None and data.labels is not None:
        if len(data.text_ids) != len(data.labels):
            msg = 'length of IDs (%d) != length of labels (%d)'
            raise ValueError(msg % (len(data.text_ids), len(data.labels)))

    if data.text_ids is not None:
        if len(data.text_ids) != rows:
            msg = 'length of IDs (%d) != number of index rows (%d)'
            raise ValueError(msg % (len(data.text_id), rows))

    if data.labels is not None:
        if len(data.labels) != rows:
            msg = 'length of labels (%d) != number of index rows (%d)'
            raise ValueError(msg % (len(data.labels), rows))

    if data.index.dtype.char in typecodes['AllFloat'] and \
            not isfinite(data.index.sum()) and \
            not isfinite(data.index).all():
        raise ValueError("index contains NaN, infinity"
                         " or a value too large for %r." % data.index.dtype)


def save_index(data, path):
    check_integrity(data)
    L.info("saving inverted index to '%s'", path)

    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_index(path):
    L.info("loading inverted index from '%s'", path)

    with open(path, 'rb') as f:
        data = pickle.load(f)

    check_integrity(data)
    return data


def save_vocabulary(vocabulary, data, path):
    L.info(" vocabulary size: %s", len(vocabulary))
    L.debug("vocabulary: %s", vocabulary.keys())
    L.info("saving vocabulary to '%s'", path)
    n_cols = get_n_cols(data)

    if len(vocabulary) != n_cols:
        msg = 'length of vocabulary (%d) != number of index columns (%d)'
        raise ValueError(msg % (len(vocabulary), n_cols))

    with open(path, 'wb') as f:
        pickle.dump(vocabulary, f)


def load_vocabulary(path, data=None):
    L.info("loading vocabulary from '%s'", path)

    with open(path, 'rb') as f:
        vocabulary = pickle.load(f)

    L.info(" vocabulary size: %s", len(vocabulary))

    if data is not None:
        n_cols = get_n_cols(data)

        if len(vocabulary) != n_cols:
            msg = 'length of vocabulary (%d) != number of index columns (%d)'
            raise ValueError(msg % (len(vocabulary), n_cols))

    return vocabulary
