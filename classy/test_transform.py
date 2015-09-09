from unittest import TestCase
from classy.transform import NGramTransformer, AnnotationTransformer, FeatureEncoder, \
    KShingleTransformer
from classy.etbase import Etc
from scipy.sparse import csr_matrix
from numpy import int32


class Sentinel(Etc):

    def __iter__(self):
        return self

    def __next__(self):
        return self


class Rows(Etc):

    def __init__(self, rows=None):
        super(Rows, self).__init__()
        self.text_columns = self.text_columns = (1,)
        self.names = ("id", "text", "attribute", "label")
        self.rows = rows

    def __iter__(self):
        if self.rows is not None:
            for row in self.rows:
                yield row
        else:
            yield [1, ["token1", "token2"], "col3", "col4"]
            yield [2, ["token3", "token4"], "col3", "col4"]


class TestNGramTransformer(TestCase):

    tokensA = [chr(i) for i in range(ord('a'), ord('d'))]
    tokensA.append('.')
    tokensB = [chr(i) for i in range(ord('A'), ord('D'))]
    tokensB.append('.')
    segments = [tokensA, tokensB]

    def test_setup(self):
        s = Sentinel()
        t = NGramTransformer(s)
        self.assertEqual(s, t.rows)
        self.assertEqual(1, t.N)
        self.assertEqual(None, t.text_columns)

    def test_unigram(self):
        expected = TestNGramTransformer.tokensA + TestNGramTransformer.tokensB
        t = NGramTransformer(Sentinel(), n=1)
        self.assertListEqual(expected,
                             list(t.n_gram(TestNGramTransformer.segments)))

    def test_bigram(self):
        expected = TestNGramTransformer.tokensA + [
            'a_b', 'b_c', 'c_.'
        ] + TestNGramTransformer.tokensB + [
            'A_B', 'B_C', 'C_.'
        ]
        t = NGramTransformer(Sentinel(), n=2)
        self.assertListEqual(expected,
                             list(t.n_gram(TestNGramTransformer.segments)))

    def test_trigram(self):
        expected = TestNGramTransformer.tokensA + [
            'a_b', 'b_c', 'c_.'
        ] + [
            'a_b_c', 'b_c_.'
        ] + TestNGramTransformer.tokensB + [
            'A_B', 'B_C', 'C_.'
        ] + [
            'A_B_C', 'B_C_.'
        ]
        t = NGramTransformer(Sentinel(), n=3)
        self.assertListEqual(expected,
                             list(t.n_gram(TestNGramTransformer.segments)))

    def test_extract(self):
        row = [TestNGramTransformer.segments, []]
        expected = [TestNGramTransformer.tokensA +
                    TestNGramTransformer.tokensB, ['~void~']]
        t = NGramTransformer(Sentinel(), n=1)

        for i in range(len(row)):
            t._extract(row, i)

        self.assertListEqual(expected, row)

    def test_iter(self):
        expected = [1, TestNGramTransformer.tokensA +
                    TestNGramTransformer.tokensB]
        n = -1
        rows = Rows([[1, TestNGramTransformer.segments]] * 3)

        for n, row in enumerate(NGramTransformer(rows, n=1)):
            self.assertListEqual(expected, row)

        self.assertEqual(2, n)


class TestKShingleTransformer(TestCase):

    tokensA = [chr(i) for i in range(ord('a'), ord('d'))]
    tokensA.append('.')
    tokensB = [chr(i) for i in range(ord('A'), ord('D'))]
    tokensB.append('.')
    segments = tokensA + tokensB

    def test_setup(self):
        s = Sentinel()
        t = KShingleTransformer(s)
        self.assertEqual(s, t.rows)
        self.assertEqual(1, t.K)
        self.assertEqual(None, t.text_columns)

    def test_unishingle(self):
        expected = []
        t = KShingleTransformer(Sentinel(), k=1)
        self.assertListEqual(expected,
                             list(t.k_shingle(TestKShingleTransformer.segments)))

    def test_bishingle(self):
        expected = ['.+A', '.+B', '.+C', '.+a', '.+b', '.+c',
                    'A+B', 'A+C', 'A+a', 'A+b', 'A+c',
                    'B+C', 'B+a', 'B+b', 'B+c',
                    'C+a', 'C+b', 'C+c',
                    'a+b', 'a+c',
                    'b+c']
        t = KShingleTransformer(Sentinel(), k=2)
        self.assertListEqual(expected,
                             list(t.k_shingle(TestKShingleTransformer.segments)))

    def test_trishingle(self):
        expected = ['.+A', '.+B', '.+C', '.+a', '.+b', '.+c',
                    'A+B', 'A+C', 'A+a', 'A+b', 'A+c',
                    'B+C', 'B+a', 'B+b', 'B+c',
                    'C+a', 'C+b', 'C+c',
                    'a+b', 'a+c', 'b+c',
                    '.+A+B', '.+A+C', '.+A+a', '.+A+b', '.+A+c',
                    '.+B+C', '.+B+a', '.+B+b', '.+B+c',
                    '.+C+a', '.+C+b', '.+C+c',
                    '.+a+b', '.+a+c', '.+b+c',
                    'A+B+C', 'A+B+a', 'A+B+b', 'A+B+c', 'A+C+a',
                    'A+C+b', 'A+C+c', 'A+a+b', 'A+a+c', 'A+b+c',
                    'B+C+a', 'B+C+b', 'B+C+c', 'B+a+b', 'B+a+c', 'B+b+c',
                    'C+a+b', 'C+a+c', 'C+b+c',
                    'a+b+c']
        t = KShingleTransformer(Sentinel(), k=3)
        self.assertListEqual(expected,
                             list(t.k_shingle(TestKShingleTransformer.segments)))

    def test_iter(self):
        expected = [1, TestKShingleTransformer.tokensA +
                    TestKShingleTransformer.tokensB]
        n = -1
        rows = Rows([[1, TestKShingleTransformer.segments]] * 3)

        for n, row in enumerate(KShingleTransformer(rows)):
            self.assertListEqual(expected, row)

        self.assertEqual(2, n)

class TestAnnotationTransformer(TestCase):

    groups = {1: (2, 3)}

    def test_init(self):
        rows = Rows()
        groups = TestAnnotationTransformer.groups
        at = AnnotationTransformer(rows, groups)
        self.assertDictEqual(groups, at.groups)

    def test_iter(self):
        rows = Rows()
        groups = TestAnnotationTransformer.groups
        at = AnnotationTransformer(rows, groups)
        at = iter(at)
        row = next(at)
        self.assertListEqual([1, ['token1', 'token2', 'attribute#col3:token1',
                                  'attribute#col3:token2', 'label#col4:token1',
                                  'label#col4:token2'], 'col3', 'col4'], row)
        row = next(at)
        self.assertListEqual([2, ['token3', 'token4', 'attribute#col3:token3',
                                  'attribute#col3:token4', 'label#col4:token3',
                                  'label#col4:token4'], 'col3', 'col4'], row)

    def test_index_error_annotation_col(self):
        rows = Rows()
        groups = {1: (3, 4)}
        at = AnnotationTransformer(rows, groups)
        self.assertRaisesRegex(RuntimeError,
                               r'len\(row\)=4, but annotation_col_indices=\(3, 4\)',
                               lambda: next(iter(at)))

    def test_wrong_annotation_col(self):
        rows = Rows()
        groups = {1: (1, 3)}
        at = AnnotationTransformer(rows, groups)
        self.assertRaisesRegex(RuntimeError,
                               r'not all annotation_columns=\(1, 3\) are strings',
                               lambda: next(iter(at)))

    def test_index_error_text_col(self):
        rows = Rows()
        groups = {4: (2, 3)}
        self.assertRaisesRegex(AssertionError,
                               r'column 4 \[ERROR\] not a known token column: \(1,\)',
                               AnnotationTransformer, rows, groups)

    def test_wrong_text_col(self):
        rows = Rows()
        groups = {2: (3,)}
        self.assertRaisesRegex(AssertionError,
                               r'column 2 \[attribute\] not a known token column: \(1,\)',
                               AnnotationTransformer, rows, groups)


class TestFeatureEncoder(TestCase):

    rows = [
        ["ID 1", ["a", "b", "c", "a", "b", "a"], ["a", "b", "x"], "Label 1"],
        ["ID 2", ["a", "b", "d", "a", "b", "a"], ["a", "x", "d"], "Label 2"],
    ]

    def test_init(self):
        rows = Rows(TestFeatureEncoder.rows)
        fe = FeatureEncoder(rows)
        self.assertEqual(None, fe.vocabulary)
        self.assertEqual(0, fe.id_col)
        self.assertEqual(-1, fe.label_col)

    def test_multirow_token_generator(self):
        rows = Rows(TestFeatureEncoder.rows)
        rows.names = ("id", "text1", "text2", "label")
        rows.text_columns = (1, 2)
        fe = FeatureEncoder(rows)

        for instance in range(2):
            expected = []
            for i in range(1, 3):
                for t in rows.rows[instance][i]:
                    expected.append('text{}={}'.format(i, t))
            gen = fe._multirow_token_generator(rows.rows[instance])
            received = []

            for idx, token in enumerate(gen):
                self.assertEqual(expected[idx], token)
                received.append(token)

            self.assertListEqual(expected, received)

    def test_sparse_matrix(self):
        rows = Rows(TestFeatureEncoder.rows)
        fe = FeatureEncoder(rows)
        inv_idx = fe.make_sparse_matrix()
        expected = csr_matrix(((3, 2, 1, 3, 2, 1),
                               ((0, 0, 0, 1, 1, 1), (0, 1, 2, 0, 1, 3))),
                              shape=(2, 4),
                              dtype=int32)
        to_string = lambda mat: str(mat.toarray()).replace('\n', ',')
        # scipy.sparse.matrix.nnz: number of non-zero values
        self.assertEqual(abs(inv_idx - expected).nnz, 0,
                         "{} != {}".format(to_string(expected),
                                           to_string(inv_idx)))

    def test_make_vocabulary(self):
        rows = Rows(TestFeatureEncoder.rows)
        fe = FeatureEncoder(rows)
        fe.make_sparse_matrix()
        self.assertEqual({'a': 0, 'b': 1, 'c': 2, 'd': 3}, fe.vocabulary)
