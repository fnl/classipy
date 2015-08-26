"""
.. py:module:: classy.generate
   :synopsis: Evaluate a classifier against an inverted index.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging
from classy.classifiers import build
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

L = logging.getLogger(__name__)

def evaluate_model(args):
    if not args.model:
        pipeline = []
        classy, params = build(args.classifier)

        if args.tfidf:
            pipeline.append(('transform', tfidf_transform(params)))

        pipeline.append(('classifier', classy))
        pipeline = Pipeline(pipeline)
    else:
        pipeline = joblib.load(args.model)



    L.debug("%s", args)
