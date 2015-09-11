"""
.. py:module:: classy.learn
   :synopsis: Training a text classifier model.

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
    report_features(args, data, pipeline)


def make_pipeline(args):
    data = load_index(args.index)

    if data.labels is None or len(data.labels) == 0:
        raise RuntimeError("input data has no labels to learn from")

    pipeline = []
    presets = make_presets(args)
    classifier, parameters = build(args.classifier, args.jobs,
                                   presets['classify'])

    if hasattr(args, "grid_search") and args.grid_search:
        L.debug("filtering zero variance features "
                "to protect from divisions by zero during grid-search")
        pipeline.append(('filter', VarianceThreshold(**presets['filter'])))

    if args.tfidf:
        L.debug("transforming features with TF-IDF")
        tfidf, params = tfidf_transform(**presets['transform'])
        pipeline.append(('transform', tfidf))
        parameters.update(params)

    if args.scale:
        L.debug("scaling all features to norm")
        pipeline.append(('scale', Normalizer(**presets['scale'])))
        parameters['scale__norm'] = ['l1', 'l2']

    if args.filter:
        L.debug("filtering features with L1-penalized %s",
                "Logistic Regression" if args.classifier == "svm" else
                "Linear SVM")
        selector, params = l1_selector(args, presets)
        pipeline.append(('select', selector))
        parameters.update(params)

    pipeline.append(('classify', classifier))
    return Pipeline(pipeline), parameters, data


def make_presets(args):
    presets = {'classify': {},
               'filter': {},
               'select': {},
               'scale': {},
               'transform': {}}

    if hasattr(args, "parameters") and args.parameters:
        for param in args.parameters.split(','):
            key, value = param.split('=', 1)
            name, prop = key.split('__', 1)
            name = name.strip()
            prop = prop.strip()
            value = eval(value)
            L.debug('preset for %s: %s=%s', name, prop, repr(value))

            try:
                presets[name][prop] = value
            except KeyError:
                L.error('"%s" is not a valid preset group name', name)

    return presets


def l1_selector(args, presets):
    presets['select']['penalty'] = 'l1'

    if args.classifier in ("svm", "sgd", "rbf"):
        selector = maxent
    else:
        selector = svm

        if 'loss' not in presets['select']:
            presets['select']['loss'] = 'squared_hinge'

        if 'dual' not in presets['select']:
            presets['select']['dual'] = False

    model, params = selector(**presets['select'])
    # C's <1 lead to empty feature sets -> too much "pruning"!
    clean_params = dict(select__C=[1e4, 1e2, 1])

    for key, values in params.items():
        if not (key.endswith('__penalty') or key.endswith('__loss') or
                key.endswith('__C')):
            clean_params[key.replace('classify', 'select')] = values

    return model, clean_params


def grid_search(pipeline, params, data, jobs=-1):
    L.debug("grid-search: jobs=%s, parameters: %s", jobs, params)
    grid = GridSearchCV(pipeline, params, scoring=Scorer,
                        cv=4, refit=True, n_jobs=jobs, verbose=1)
    grid.fit(data.index, data.labels)
    print("Best score:", grid.best_score_)
    print("Parameters:")

    for name, value in grid.best_params_.items():
        print('{}={}'.format(name, repr(value)))

    return grid.best_estimator_


def report_features(args, data, pipeline):
    classifier = pipeline._final_estimator

    if hasattr(classifier, "coef_"):
        n_coeffs = classifier.coef_.shape[1]
        L.debug("number of final classifier coefficients: %s", n_coeffs)

        if args.vocabulary and not args.filter:
            cov = make_inverted_vocabulary(args.vocabulary, data)

            if len(cov) == n_coeffs:
                print_top_features(classifier, data, cov)
            else:
                L.info("vocab. size and classif. coefficients vector length "
                       "do not match (voc=%s vs coeff=%s)", len(cov), n_coeffs)


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
