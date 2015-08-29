"""
.. py:module:: classy.classifiers
   :synopsis: All classifiers available to classy, with their parameter ranges for a grid search.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC

CLASSIFIERS = {}


def build(option, data, jobs=-1):
    try:
        classy = CLASSIFIERS[option]()
    except KeyError:
        msg = 'unknown classifier {}'
        raise RuntimeError(msg.format(option))

    if len(data.label_names) > 2:
        return OneVsRestClassifier(classy, n_jobs=jobs)
    else:
        return classy


def ridge():
    return RidgeClassifier(max_iter=1e4, class_weight='auto'), {
        'classifier__alpha': [100., 10., 1., .1, .01, .001],
        'classifier__normalize': [True, False],
        #'classifier__class_weight': ['auto', None],
        #'classifier__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg'],
        'classifier__tol': [.01, 1e-3, 1e-6],
    }

CLASSIFIERS[ridge.__name__] = ridge


def svm():
    return LinearSVC(loss='hinge', max_iter=1e4, class_weight='auto'), {
        # Hinge loss is generally considered the better choice for sparse data
        'classifier__C': [1e8, 1e4, 1e2, 1e0, 1e-2, 1e-4],
        'classifier__loss': ['hinge', 'squared_hinge'],
        # 'classifier__dual': [True, False],  # doesn't mix...
        #'classifier__class_weight': ['auto', None],
        'classifier__intercept_scaling': [10., 1., .1],
        'classifier__tol': [.01, 1e-4, 1e-8],
    }

CLASSIFIERS[svm.__name__] = svm


def maxent():
    return LogisticRegression(max_iter=1e4, class_weight='auto'), {
        'classifier__C': [1e8, 1e4, 1e2, 1e0, 1e-2, 1e-4],
        #'classifier__class_weight': ['auto', None],
        'classifier__intercept_scaling': [10., 1., .1],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__tol': [.01, 1e-4, 1e-8],
    }

CLASSIFIERS[maxent.__name__] = maxent


def multinomial():
    return MultinomialNB(), {
        'classifier__alpha': [10., 1., 0., .1, .01, .001],
        'classifier__fit_prior': [True, False],
    }

CLASSIFIERS[multinomial.__name__] = multinomial


def bernoulli():
    return BernoulliNB(), {
        'classifier__alpha': [10., 1., 0., .1, .01, .001],
        'classifier__fit_prior': [True, False],
    }

CLASSIFIERS[bernoulli.__name__] = bernoulli