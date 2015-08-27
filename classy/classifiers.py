"""
.. py:module:: classy.classifiers
   :synopsis: All classifiers available to classy.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC

CLASSIFIER = {}


def build(option, data):
    try:
        classy = CLASSIFIER[option]()
    except KeyError:
        msg = 'unknown classifier {}'
        raise RuntimeError(msg.format(option))

    if len(data.label_names) > 2:
        return OneVsRestClassifier(classy, n_jobs=-1)
    else:
        return classy


def ridge():
    return RidgeClassifier(), {
        'classifier__alpha': [10., 1., 0., .1, .01],
        'classifier__normalize': [True, False],
        'classifier__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg'],
        'classifier__tol': [.1, .01, 1e-3, 1e-6],
    }

CLASSIFIER[ridge.__name__] = ridge


def svm():
    return LinearSVC(loss='hinge'), {
        # Hinge loss is generally considered the better choice for sparse data
        'classifier__C': [100., 10., 1., .1, .01],
        'classifier__class_weight': ['auto', None],
        'classifier__intercept_scaling': [10., 5., 1., .5],
        'classifier__tol': [.1, .01, 1e-4, 1e-8],
    }

CLASSIFIER[svm.__name__] = svm


def maxent():
    return LogisticRegression(class_weight='auto'), {
        'classifier__C': [100., 10., 1., .1, .01],
        'classifier__class_weight': ['auto', None],
        'classifier__intercept_scaling': [10., 5., 1., .5],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__tol': [.1, .01, 1e-4, 1e-8],
    }

CLASSIFIER[maxent.__name__] = maxent


def multinomial():
    return MultinomialNB(), {
        'classifier__alpha': [10., 1., 0., .1, .01],
        'classifier__fit_prior': [True, False],
    }

CLASSIFIER[multinomial.__name__] = multinomial


def bernoulli():
    return BernoulliNB(), {
        'classifier__alpha': [10., 1., 0., .1, .01],
        'classifier__binarize': [True, False],
        'classifier__fit_prior': [True, False],
    }

CLASSIFIER[bernoulli.__name__] = bernoulli