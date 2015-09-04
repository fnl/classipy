"""
.. py:module:: classy.classifiers
   :synopsis: All classifiers available to classy, with their parameter ranges for a grid search.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
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
        'classifier__alpha': [1e3, 10, 1, .1, 1e-3],
        'classifier__normalize': [True, False],
        'classifier__tol': [.05, 1e-3, 1e-6],
        # 'classifier__class_weight': ['auto', None],
        # 'classifier__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg'],
    }

CLASSIFIERS[ridge.__name__] = ridge


def svm():
    return LinearSVC(loss='hinge', max_iter=1e4, class_weight='auto'), {
        'classifier__C': [1e8, 1e3, 1, 1e-2, 1e-4],
        'classifier__loss': ['hinge', 'squared_hinge'],
        'classifier__intercept_scaling': [10., 1., .1],
        'classifier__tol': [.05, 1e-4, 1e-8],
        # 'classifier__class_weight': ['auto', None],
        # 'classifier__dual': [True, False],  # doesn't mix...
    }

CLASSIFIERS[svm.__name__] = svm


def maxent():
    return LogisticRegression(max_iter=1e4, class_weight='auto'), {
        'classifier__C': [1e8, 1e3, 1, 1e-2, 1e-4],
        'classifier__intercept_scaling': [10, 1, .1],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__tol': [.05, 1e-4, 1e-8],
        # 'classifier__class_weight': ['auto', None],
    }

CLASSIFIERS[maxent.__name__] = maxent


def randomforest():
    return RandomForestClassifier(n_estimators=40, max_depth=25,
                                  criterion='gini', max_features='auto',
                                  class_weight='auto', oob_score=False), {
        'classifier__n_estimators': [80, 40, 20, 10],
        'classifier__max_depth': [100, 25, 5, 2],
        # 'classifier__class_weight': ['auto', None],
        # 'classifier__max_features': ['sqrt', None],
        # 'classifier__criterion': ['gini', 'entropy'],
        # 'classifier__oob_score': [True, False],  # breaks: bug #4954
    }

CLASSIFIERS[randomforest.__name__] = randomforest


def multinomial():
    return MultinomialNB(), {
        # NB: class_prior=None is like class_weight='auto'
        'classifier__alpha': [10., 1., 0., .1, .01, .001],
        'classifier__fit_prior': [True, False],
    }

CLASSIFIERS[multinomial.__name__] = multinomial


def bernoulli():
    return BernoulliNB(), {
        # NB: class_prior=None is like class_weight='auto'
        'classifier__alpha': [10., 1., 0., .1, .01, .001],
        'classifier__fit_prior': [True, False],
    }

CLASSIFIERS[bernoulli.__name__] = bernoulli