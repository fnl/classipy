"""
.. py:module:: classy.generate
   :synopsis: Evaluate a classifier against an inverted index.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""

import logging

L = logging.getLogger(__name__)

def evaluate_model(args):
    L.debug("%s", args)
