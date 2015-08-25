"""
.. py:module:: classy.learn
   :synopsis: Train a text classifier.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
import pickle
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

L = logging.getLogger(__name__)
CLASSIFIER = {}

# A scoring function that is robust against class-imbalances.
Scorer = metrics.make_scorer(metrics.matthews_corrcoef)


def learn_model(args):
    L.debug("%s", args)
    pipeline = []

    try:
        classy, params = CLASSIFIER[args.classifier]()
    except KeyError:
        msg = 'unknown classifier {}'.format(args.classifier)
        raise RuntimeError(msg)

    if args.tfidf:
        tfidf = TfidfTransformer(norm='l2', sublinear_tf=True,
                                 smooth_idf=True, use_idf=True)
        pipeline.append(('transform', tfidf))
        params.update({
            'transform__norm': [None, 'l1', 'l2'],
            'transform__use_idf': [True, False],
            'transform__smooth_idf': [True, False],
            'transform__sublinear_tf': [True, False],
        })

    pipeline.append(('classifier', classy))
    pipeline = Pipeline(pipeline)

    with open(args.index, 'rb') as f:
        labels, index = pickle.load(f)

    if args.grid_search:
        grid = GridSearchCV(pipeline, params, scoring=Scorer,
                            cv=5, refit=False, n_jobs=4, verbose=1)
        grid.fit(index, labels)

        print("best score:", grid.best_score_)

        for name, value in grid.best_params_.items():
            print('{}:\t{}'.format(name, repr(value)))

        L.warn("grid search result not yet written as model")
    else:
        pipeline.fit(index, labels)
        joblib.dump(pipeline, args.model)


def ridge():
    return RidgeClassifier(), {
        'classifier__alpha': [10., 1., 0., .1, .01],
        'classifier__normalize': [True, False],
        'classifier__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg'],
        'classifier__tol': [.1, .01, 1e-3, 1e-6],
    }

CLASSIFIER[ridge.__name__] = ridge


def svm():
    return LinearSVC(loss='hinge'), {  # Hinge loss is better for sparse data
        'classifier__C': [100., 10., 1., .1, .01],
        'classifier__class_weight': ['auto', None],
        'classifier__intercept_scaling': [10., 5., 1., .5],
        'classifier__penalty': ['hinge', 'squared_hinge'],
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
