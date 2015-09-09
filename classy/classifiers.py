"""
.. py:module:: classy.classifiers
   :synopsis: All classifiers available to classy,
              with generic, tuned parameter ranges for the grid search.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC


CLASSIFIERS = {}


def build(option, data, jobs=-1):
    """
    Create a classifier and add a one-vs-rest wrapper around it
    if the data is multinomial.

    :return: A (classifier instance, parameter dictionary) tuple
    """
    try:
        classy = CLASSIFIERS[option]()
    except KeyError:
        msg = 'unknown classifier {}'
        raise RuntimeError(msg.format(option))

    if len(data.label_names) > 2:
        return OneVsRestClassifier(classy, n_jobs=jobs)
    else:
        return classy


def tfidf_transform(params):
    tfidf = TfidfTransformer(
        norm='l2', sublinear_tf=True, smooth_idf=True, use_idf=True
    )
    params.update({
        'transform__norm': ['l1', 'l2'],
        'transform__sublinear_tf': [True, False],
        # 'transform__use_idf': [True, False],
        # Lidstone-like smoothing cannot be disabled:
        # 'transform__smooth_idf': [True, False],  # divides by zero...
    })
    return tfidf


def ridge():
    return RidgeClassifier(max_iter=1e4, class_weight='auto', solver='auto'), {
        'classify__alpha': [1e3, 1e1, 1, 1e-1, 1e-2],
        'classify__normalize': [True, False],
        'classify__tol': [.05, 1e-3, 1e-6],
        # 'classify__class_weight': ['auto', None],
        # 'classify__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg'],
    }

CLASSIFIERS[ridge.__name__] = ridge


def svm(loss='hinge', max_iter=1e4, class_weight='auto', **keys):
    return LinearSVC(loss=loss, max_iter=max_iter, class_weight=class_weight,
                     **keys), {
        'classify__C': [1e5, 1e2, 1, 1e-1, 1e-2],
        'classify__loss': ['hinge', 'squared_hinge'],
        'classify__intercept_scaling': [10., 1., .1],
        'classify__tol': [.05, 1e-4, 1e-8],
        # 'classify__class_weight': ['auto', None],
        # 'classify__dual': [True, False],  # doesn't mix...
    }

CLASSIFIERS[svm.__name__] = svm


def rbf(max_iter=1e4, class_weight='auto', **keys):
    return SVC(max_iter=max_iter, class_weight=class_weight, **keys), {
        'classify__C': [1e5, 1e2, 1, 1e-1, 1e-2],
        'classify__tol': [.05, 1e-4, 1e-8],
        # 'classify__class_weight': ['auto', None],
    }

CLASSIFIERS[rbf.__name__] = rbf


def maxent(max_iter=1e4, class_weight='auto', **keys):
    return LogisticRegression(max_iter=max_iter, class_weight=class_weight,
                              **keys), {
        'classify__C': [1e5, 1e2, 1, 1e-1, 1e-2],
        'classify__intercept_scaling': [10, 1, .1],
        'classify__penalty': ['l1', 'l2'],
        'classify__tol': [.05, 1e-4, 1e-8],
        # 'classify__class_weight': ['auto', None],
    }

CLASSIFIERS[maxent.__name__] = maxent


def randomforest():
    return RandomForestClassifier(n_estimators=40, max_depth=25,
                                  criterion='gini', max_features='auto',
                                  class_weight='auto', oob_score=False), {
        'classify__n_estimators': [80, 40, 20, 10],
        'classify__max_depth': [100, 25, 5, 2],
        # 'classify__class_weight': ['auto', None],
        # 'classify__max_features': ['sqrt', None],
        # 'classify__criterion': ['gini', 'entropy'],
        # 'classify__oob_score': [True, False],  # breaks: bug #4954
    }

CLASSIFIERS[randomforest.__name__] = randomforest


def multinomial():
    return MultinomialNB(), {
        # NB: class_prior=None is like class_weight='auto'
        'classify__alpha': [10., 1., 0., .1, .01, .001],
        'classify__fit_prior': [True, False],
    }

CLASSIFIERS[multinomial.__name__] = multinomial


def bernoulli():
    return BernoulliNB(), {
        # NB: class_prior=None is like class_weight='auto'
        'classify__alpha': [10., 1., 0., .1, .01, .001],
        'classify__fit_prior': [True, False],
    }

CLASSIFIERS[bernoulli.__name__] = bernoulli