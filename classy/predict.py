"""
.. py:module:: classy.predict
   :synopsis: Predict labels for an inverted index.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging

L = logging.getLogger(__name__)

def predict_labels(args):
    L.debug("%s", args)
