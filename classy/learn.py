"""
.. py:module:: classy.learn
   :synopsis: Train a text classifier.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
from classy.classifiers import build
from classy.data import load_index
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

L = logging.getLogger(__name__)

# A scoring function that is robust against class-imbalances.
Scorer = metrics.make_scorer(metrics.matthews_corrcoef)


def learn_model(args):
    L.debug("%s", args)
    pipeline = []
    classy, params = build(args.classifier)

    if args.tfidf:
        pipeline.append(('transform', tfidf_transform(params)))

    pipeline.append(('classifier', classy))
    pipeline = Pipeline(pipeline)
    doc_ids, labels, index = load_index(args.index)

    if args.grid_search:
        pipeline = grid_search(pipeline, params, index, labels)

    pipeline.fit(index, labels)
    joblib.dump(pipeline, args.model)


def grid_search(pipeline, params, index, labels):
    grid = GridSearchCV(pipeline, params, scoring=Scorer,
                        cv=5, refit=True, n_jobs=4, verbose=1)
    grid.fit(index, labels)
    print("Best score:", grid.best_score_)
    print("Parameters:")

    for name, value in grid.best_params_.items():
        print('{} = {}'.format(name, repr(value)))

    return grid.best_estimator_


def tfidf_transform(params):
    tfidf = TfidfTransformer(
        norm='l2', sublinear_tf=True, smooth_idf=True, use_idf=True
    )
    params.update({
        'transform__norm': [None, 'l1', 'l2'],
        'transform__use_idf': [True, False],
        'transform__sublinear_tf': [True, False],
        # Lidstone-like smoothing cannot be disabled
        # (a SciKit-Learn bug similar to #3637?)
        # 'transform__smooth_idf': [True, False],
    })
    return tfidf
