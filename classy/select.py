"""
.. py:module:: classy.select
   :synopsis: Feature selection over the inverted indices.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
from sklearn.feature_selection import SelectKBest, chi2, RFE
from numpy import diff, ones, cumsum, where
from scipy.sparse import csc_matrix
from .data import load_vocabulary, save_index, load_index, save_vocabulary, get_n_cols
from sklearn.svm import LinearSVC


L = logging.getLogger(__name__)


def select_features(args):
    L.debug("%s", args)
    data = load_index(args.index)

    if args.vocabulary:
        vocabulary = load_vocabulary(args.vocabulary)
    else:
        vocabulary = None

    if args.cutoff > 1:
        data = drop_words(args.cutoff, data, vocabulary)

    if args.select > 0:
        data = select_best(args.select, data, vocabulary)

    if args.eliminate > 0:
        data = eliminate_words(args.eliminate, data, vocabulary)

    save_index(data, args.new_index)

    if vocabulary and args.new_vocabulary:
        save_vocabulary(vocabulary, data, args.new_vocabulary)


def drop_words(min_df, data, vocabulary=None):
    """
    Prune words below some minimum document frequency ``min_df`` from the
    vocabulary (in-place) and drop those columns from the inverted index.

    :param vocabulary: vocabulary dictionary (will be updated in-place)
    :param data: ``Data`` structure
    :param min_df: integer; minimum document frequency
    :return: a new ``Data`` structure
    """
    L.debug("dropping features below df=%s", min_df)
    df = diff(csc_matrix(data.index, copy=False).indptr)  # document frequency
    mask = ones(len(df), dtype=bool)  # mask: columns that can/cannot stay
    mask &= df >= min_df  # create a "mask" of columns above cutoff
    keep = where(mask)[0]  # determine which columns to keep
    data = data._replace(index=data.index[:, keep])  # drop unused columns
    prune(vocabulary, mask)
    return data


def eliminate_words(k, data, vocabulary=None, steps=100):
    L.debug("recursively eliminating features down to %s", k)
    estimator = LinearSVC(penalty='l2', loss='hinge', class_weight='auto')
    step = (get_n_cols(data) - k) // steps + 1
    selector = RFE(estimator, k, step=step)
    selector = selector.fit(data.index, data.labels)
    mask = selector.support_
    data = data._replace(index=data.index[:, mask])
    prune(vocabulary, mask)
    return data


def select_best(k, data, vocabulary=None):
    """
    Select the top ``k`` most informative words (using a chi-square test)
    and drop everything else from the index and vocabulary.

    :param k: integer; the most informative features to maintain
    :param data: ``Data`` structure
    :param vocabulary: vocabulary dictionary (will be updated in-place)
    :return: a new ``Data`` structure
    """
    L.debug("selecting K=%s best features", k)
    selector = SelectKBest(chi2, k=k)
    selector.fit(data.index, data.labels)
    mask = selector._get_support_mask()
    data = data._replace(index=data.index[:, mask])
    prune(vocabulary, mask)
    return data


def prune(vocabulary, mask):
    """Clean up the vocabulary with the given mask and new indices."""
    if vocabulary:
        new_idx = cumsum(mask) - 1  # new indices (with array len as old)

        for word in list(vocabulary.keys()):
            idx = vocabulary[word]

            if mask[idx]:
                # noinspection PyUnresolvedReferences
                vocabulary[word] = new_idx[idx]
            else:
                del vocabulary[word]

        L.debug("vocabulary - new size: %s", len(vocabulary))
