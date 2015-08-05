"""
.. py:module:: 
   :synopsis: 

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""
import logging
import itertools


class Transformer:

   def __init__(self, extractor, N=2, K=1):
       self.rows = iter(extractor)
       self.N = N
       self.K = K
       self._token_cols = []

   def __iter__(self):
       self._token_cols = []
       row = next(self.rows)

       for i, v in enumerate(row):
           if isinstance(v, list):
              self._token_cols.append(i)
              self._extract(row, i)

       if len(self._token_cols) == 0:
           logging.warning('Transformer found no token columns in input')

       yield row

       for row in self.rows:
           for i in self._token_cols:
               self._extract(row, i)

           yield row

   def _extract(self, row, i):
       if row[i] and len(row[i]):
           ngrams = self.ngram(row[i])
           shingles = self.kshingle(row[i])
           row[i] = list(itertools.chain(ngrams, shingles))

   def ngram(self, token_segments):
       N = range(1, self.N + 1)

       for segment in token_segments:
           for n in N:
               for i in range(len(segment) - n + 1):
                   yield " ".join(segment[i:i+n])

   def kshingle(self, token_segments):
       words = sorted(set(w for s in token_segments for w in s if w.isalnum()))
       words = list(words)

       for k in range(2, self.K + 1):
           for shingle in itertools.combinations(words, k):
               yield "_".join(sorted(shingle))


class FeatureEncoder:

   def __init__(self, transformer):
      self.rows = transformer
      self.feature_cols = None

   def __iter__(self):
      return self

   def __next__(self):
      row = next(self.rows)

      if self.feature_cols is None:
         self.feature_cols = []
         found_text = False

         for i, col in row:
            if isinstance(col, list):
               found_text = True
            elif found_text:
               self.feature_cols.append((i, []))

      for i, features in self.feature_cols:
         idx = features.index(row[i])

         if idx == -1:
            idx = len(features)
            features.append(row[i])



