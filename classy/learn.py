"""
.. py:module:: classy.learn
   :synopsis: Train a text classifier.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
from classy.classifiers import build
from classy.data import load_index, load_vocabulary
from numpy import argsort, array
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
    pipeline, parameters, data = make_pipeline(args)

    if args.grid_search:
        pipeline = grid_search(pipeline, parameters, data, args.jobs)

    pipeline.fit(data.index, data.labels)
    joblib.dump(pipeline, args.model)

    if args.vocabulary:
        voc = load_vocabulary(args.vocabulary, data)
        cov = array(list(voc.keys()))

        for word, idx in voc.items():
            cov[idx] = word

        classifier = pipeline._final_estimator
        L.debug("classifier coefficients shape: %s", classifier.coef_.shape)

        for i in range(classifier.coef_.shape[0]):
            top_n = argsort(classifier.coef_[i])[:10]
            worst_n = argsort(classifier.coef_[i])[-10:][::-1]
            print('label {2} features (top-worst): "{0}", ... "{1}"'.format(
                '", "'.join(cov[top_n]),
                '", "'.join(cov[worst_n]), data.label_names[i],
            ))


def make_pipeline(args):
    pipeline = []
    data = load_index(args.index)

    if data.labels is None or len(data.labels) == 0:
        raise RuntimeError("input data has no labels to learn from")

    classifier, parameters = build(args.classifier, data, args.jobs)

    if args.tfidf:
        L.debug("transforming features with TF-IDF")
        pipeline.append(('transform', tfidf_transform(parameters)))

    pipeline.append(('classifier', classifier))
    pipeline = Pipeline(pipeline)

    return pipeline, parameters, data


def grid_search(pipeline, params, data, jobs=-1):
    grid = GridSearchCV(pipeline, params, scoring=Scorer,
                        cv=5, refit=True, n_jobs=jobs, verbose=1)
    grid.fit(data.index, data.labels)
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
        # Lidstone-like smoothing cannot be disabled (divide by zero)
        # 'transform__smooth_idf': [True, False],
    })
    return tfidf
