"""
.. py:module:: classy.learn
   :synopsis: Train a text classifier.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
from numpy import argsort
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from .classifiers import build, tfidf_transform, maxent, svm
from .data import load_index, make_inverted_vocabulary
from sklearn.preprocessing import Normalizer


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
        cov = make_inverted_vocabulary(args.vocabulary, data)
        classifier = pipeline._final_estimator
        L.debug("classifier coefficients shape: %s", classifier.coef_.shape)
        print_top_features(classifier, data, cov)


def make_pipeline(args):
    pipeline = []
    data = load_index(args.index)
    presets = {'classify': {},
               'filter': {},
               'select': {},
               'scale': {},
               'transform': {}}

    if data.labels is None or len(data.labels) == 0:
        raise RuntimeError("input data has no labels to learn from")

    if args.parameters:
        for param in args.parameters.split(','):
            key, value = param.split('=', 1)
            name, prop = key.split('__', 1)
            value = eval(value)
            L.debug('preset for %s: %s=%s', name, prop, repr(value))

            try:
                presets[name][prop] = value
            except KeyError:
                L.error('%s not a valid preset group name', name)

    classifier, parameters = build(args.classifier, data, args.jobs, presets['classify'])

    if hasattr(args, "grid_search") and args.grid_search:
        L.debug("filtering zero variance features "
                "to protect from divisions by zero")
        pipeline.append(('filter', VarianceThreshold(**presets['filter'])))

    if args.extract:
        L.debug("extracting features with %s",
                "Logistic Regression" if args.classifier == "svm" else
                "a linear SVM")

        if 'penalty' not in presets['select']:
            presets['select']['penalty'] = 'l1'

        if args.classifier == "svm":
            selector = maxent
        else:
            if 'loss' not in presets['select']:
                presets['select']['loss'] = 'squared_hinge'

            if 'dual' not in presets['select']:
                presets['select']['dual'] = False

            selector = svm

        model, params = selector(**presets['select'])
        pipeline.append(('select', model))

        for key, value in params.items():
            key = key.replace('classify', 'select')

            if not (key.endswith('__penalty') or key.endswith('__loss')):
                parameters[key] = value

    if args.tfidf:
        L.debug("transforming features with TF-IDF")
        tfidf, params = tfidf_transform(**presets['transform'])
        parameters.update(params)
        pipeline.append(('transform', tfidf))

    L.debug("scaling features to norm")
    pipeline.append(('scale', Normalizer(**presets['scale'])))
    parameters['scale__norm'] = ['l1', 'l2']

    if hasattr(args, "grid_search") and args.grid_search:
        L.debug("grid-search parameters: %s", parameters)

    pipeline.append(('classify', classifier))
    pipeline = Pipeline(pipeline)

    return pipeline, parameters, data


def grid_search(pipeline, params, data, jobs=-1):
    grid = GridSearchCV(pipeline, params, scoring=Scorer,
                        cv=4, refit=True, n_jobs=jobs, verbose=1)
    grid.fit(data.index, data.labels)
    print("Best score:", grid.best_score_)
    print("Parameters:")

    for name, value in grid.best_params_.items():
        print('{} = {}'.format(name, repr(value)))

    return grid.best_estimator_


def print_top_features(classifier, data, inverted_vocabulary):
    if hasattr(classifier, "feature_importances_"):
        weights = classifier.feature_importances_
        top_n = argsort(weights)[:10]
        worst_n = argsort(weights)[-10:][::-1]
        print('features (top-worst): "{0}", ... "{1}"'.format(
            '", "'.join(inverted_vocabulary[top_n]),
            '", "'.join(inverted_vocabulary[worst_n]), data.label_names[0],
        ))
    elif hasattr(classifier, "coef_"):
        for i in range(classifier.coef_.shape[0]):
            top_n = argsort(classifier.coef_[i])[:10]
            worst_n = argsort(classifier.coef_[i])[-10:][::-1]
            print('label {2} features (top-worst): "{0}", ... "{1}"'.format(
                '", "'.join(inverted_vocabulary[top_n]),
                '", "'.join(inverted_vocabulary[worst_n]), data.label_names[i],
            ))
