"""
.. py:module:: classy.etbase
   :synopsis: Base class for all Extractors and Transformers.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""


class Etc:

    """
    Base class for all Extractors and Transformers.
    """

    def __init__(self, ante=None):
        if ante is not None:
            assert isinstance(ante, Etc)

        self._names = None if ante is None else ante.names
        self._token_columns = None if ante is None else ante.token_columns
        self._text_columns = None if ante is None else ante.text_columns

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, names):
        if names is not None:
            self._names = tuple(n for n in names if isinstance(n, str))
        else:
            self._names = None

    @property
    def token_columns(self):
        return self._token_columns

    @token_columns.setter
    def token_columns(self, token_columns):
        if token_columns is not None:
            self._token_columns = tuple(int(c) for c in token_columns)
        else:
            self._token_columns = None

    @property
    def text_columns(self):
        return self._text_columns

    @text_columns.setter
    def text_columns(self, text_columns):
        if text_columns is not None:
            self._text_columns = tuple(int(c) for c in text_columns)
        else:
            self._text_columns = None
