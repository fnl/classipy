"""
.. py:module:: classy.data
   :synopsis: Data I/O for the text classifiers.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
import pickle
from numpy.core.umath import isfinite
from numpy import typecodes, asarray, unique, where
from sklearn.preprocessing import label_binarize
from random import sample
from collections import namedtuple, Counter

L = logging.getLogger(__name__)
Data = namedtuple('Data', 'text_ids index labels label_names min_label')


def make_data(inverted_index, text_ids=None, labels=None):
    """
    Create a namedtuple "Data" with the following properties:

    - ``text_ids``: a list of document IDs (or None)
    - ``index``: the actual, inverted index (as SciPy CSR matrix)
    - ``labels``: a (binarized) array of document labels, as integers (or None)
    - ``label_names``: the list of label names per 0-based label integer
    - ``min_label``: the label integer that has the least number of examples
                     in the dataset (i.e., the likely hardest label to learn)

    :param inverted_index: the inverted index; having a ``tocsr()`` method
    :param text_ids: the list of document IDs (or None)
    :param labels: the list of document labels (or None)
    :return: a Data namedtuple
    """
    if text_ids is not None and not isinstance(text_ids, list):
        text_ids = list(text_ids)

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
    """Get the number of rows/documents in ``data``."""
    return data.index.shape[0]


def get_n_cols(data):
    """Get the number of columns/features (words) in ``data``."""
    return data.index.shape[1]


def check_integrity(data):
    """Ensure the semantic integrity of the ``data``."""
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
    """Save the ``data`` to ``path``."""
    check_integrity(data)
    L.info("saving inverted index to '%s'", path)

    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_index(path):
    """Load a Data structure from ``path``."""
    L.info("loading inverted index from '%s'", path)

    with open(path, 'rb') as f:
        data = pickle.load(f)

    check_integrity(data)
    return data


def save_vocabulary(vocabulary, data, path):
    """Save the ``vocabulary`` dict for ``data`` to ``path``."""
    assert isinstance(vocabulary, dict), "vocabulary not a dict"
    size = len(vocabulary)
    L.info("vocabulary size: %s", size)
    rnd_sample = sample(vocabulary.keys(), min(size, 10))
    L.debug("vocabulary sample: %s", ', '.join(rnd_sample))
    L.info("saving vocabulary to '%s'", path)
    n_cols = get_n_cols(data)

    if len(vocabulary) != n_cols:
        msg = 'length of vocabulary (%d) != number of index columns (%d)'
        raise ValueError(msg % (len(vocabulary), n_cols))

    with open(path, 'wb') as f:
        pickle.dump(vocabulary, f)


def load_vocabulary(path, data=None):
    """
    Load a vocabulary dict from ``path``,
    optionally ensuring it could be appropriate for ``data``.
    """
    L.info("loading vocabulary from '%s'", path)

    with open(path, 'rb') as f:
        vocabulary = pickle.load(f)

    assert isinstance(vocabulary, dict), "vocabulary not a dict"
    size = len(vocabulary)
    L.info(" vocabulary size: %s", size)
    rnd_sample = sample(vocabulary.keys(), min(size, 10))
    L.debug("vocabulary sample: %s", ', '.join(rnd_sample))

    if data is not None:
        n_cols = get_n_cols(data)

        if len(vocabulary) != n_cols:
            msg = 'length of vocabulary (%d) != number of index columns (%d)'
            raise ValueError(msg % (len(vocabulary), n_cols))

    return vocabulary
