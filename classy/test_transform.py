from unittest import TestCase
from classy.transform import Transformer, AnnotationTransformer
from testfixtures import LogCapture


class Sentinel:
    def __iter__(self):
        return self
    def __next__(self):
        return self

class TestTransformer(TestCase):

    tokensA = [chr(i) for i in range(ord('a'), ord('d'))]
    tokensA.append('.')
    tokensB = [chr(i) for i in range(ord('A'), ord('D'))]
    tokensB.append('.')
    segments = [tokensA, tokensB]

    def test_setup(self):
        s = Sentinel()
        t = Transformer(s)
        self.assertEqual(s, t.rows)
        self.assertEqual(2, t.N)
        self.assertEqual(1, t.K)
        self.assertListEqual([], t._token_cols)

    def test_unigram(self):
        expected = TestTransformer.tokensA + \
                   TestTransformer.tokensB
        t = Transformer(Sentinel(), N=1)
        self.assertListEqual(expected, list(t.ngram(TestTransformer.segments)))

    def test_bigram(self):
        expected = TestTransformer.tokensA + \
                   ['a b', 'b c', 'c .'] + \
                   TestTransformer.tokensB + \
                   ['A B', 'B C', 'C .']
        t = Transformer(Sentinel(), N=2)
        self.assertListEqual(expected, list(t.ngram(TestTransformer.segments)))

    def test_trigram(self):
        expected = TestTransformer.tokensA + \
                   ['a b', 'b c', 'c .'] + \
                   ['a b c', 'b c .'] + \
                   TestTransformer.tokensB + \
                   ['A B', 'B C', 'C .'] + \
                   ['A B C', 'B C .']
        t = Transformer(Sentinel(), N=3)
        self.assertListEqual(expected, list(t.ngram(TestTransformer.segments)))

    def test_unishingle(self):
        expected = []
        t = Transformer(Sentinel(), K=1)
        self.assertListEqual(expected, list(t.kshingle(TestTransformer.segments)))

    def test_bishingle(self):
        expected = ['A_B', 'A_C', 'A_a', 'A_b', 'A_c',
                    'B_C', 'B_a', 'B_b', 'B_c',
                    'C_a', 'C_b', 'C_c',
                    'a_b', 'a_c',
                    'b_c']
        t = Transformer(Sentinel(), K=2)
        self.assertListEqual(expected, list(t.kshingle(TestTransformer.segments)))

    def test_trishingle(self):
        expected = ['A_B', 'A_C', 'A_a', 'A_b', 'A_c',
                    'B_C', 'B_a', 'B_b', 'B_c',
                    'C_a', 'C_b', 'C_c',
                    'a_b', 'a_c', 'b_c',
                    'A_B_C', 'A_B_a', 'A_B_b', 'A_B_c', 'A_C_a',
                    'A_C_b', 'A_C_c', 'A_a_b', 'A_a_c', 'A_b_c',
                    'B_C_a', 'B_C_b', 'B_C_c', 'B_a_b', 'B_a_c', 'B_b_c',
                    'C_a_b', 'C_a_c', 'C_b_c',
                    'a_b_c']
        t = Transformer(Sentinel(), K=3)
        self.assertListEqual(expected, list(t.kshingle(TestTransformer.segments)))

    def test_extract(self):
        row = [TestTransformer.segments, [], None]
        expected = [TestTransformer.tokensA +
                    TestTransformer.tokensB, [], None]
        t = Transformer(Sentinel(), N=1)

        for i in range(len(row)):
            t._extract(row, i)

        self.assertListEqual(expected, row)

    def test_iter(self):
        expected = [TestTransformer.tokensA +
                    TestTransformer.tokensB]

        n = -1

        for n, row in enumerate(Transformer([[TestTransformer.segments]]*3, N=1)):
            self.assertListEqual(expected, row)

        self.assertEqual(2, n)

    def test_warning(self):
        with LogCapture() as l:
            t = Transformer([[]]*3)
            next(iter(t))
            l.check(('root', 'WARNING',
                     'Transformer found no token columns in input'))

    def test_no_warning(self):
        with LogCapture() as l:
            t = Transformer([[TestTransformer.segments]]*3)
            next(iter(t))
            l.check()


class TestAnnotationTransformer(TestCase):

    rows = lambda: [[1, ["token1", "token2"], "col3", "col4"]]
    groups = {1: (2, 3)}

    def test_init(self):
        rows = TestAnnotationTransformer.rows()
        groups = TestAnnotationTransformer.groups
        at = AnnotationTransformer(rows, groups, 2, 3)
        self.assertTupleEqual((3, 2), at.dropped_cols)
        self.assertDictEqual(groups, at.groups)

    def test_iter(self):
        rows = TestAnnotationTransformer.rows()
        groups = TestAnnotationTransformer.groups
        at = AnnotationTransformer(rows, groups, 2, 3)
        row = next(iter(at))
        self.assertListEqual([1, ["col3:col4:token1",
                                  "col3:col4:token2"]], row)

    def test_index_error_annotation_col(self):
        rows = TestAnnotationTransformer.rows()
        groups = {1: (3, 4)}
        at = AnnotationTransformer(rows, groups)
        self.assertRaisesRegex(RuntimeError,
                               r'len\(row\)=4, but annotation_col_indices=\(3, 4\)',
                               lambda: next(iter(at)))

    def test_wrong_annotation_col(self):
        rows = TestAnnotationTransformer.rows()
        groups = {1: (1, 3)}
        at = AnnotationTransformer(rows, groups)
        self.assertRaisesRegex(RuntimeError,
                               r'not all annotation_columns=\(1, 3\) are strings',
                               lambda: next(iter(at)))

    def test_index_error_text_col(self):
        rows = TestAnnotationTransformer.rows()
        groups = {4: (2, 3)}
        at = AnnotationTransformer(rows, groups)
        self.assertRaisesRegex(RuntimeError,
                               r'len\(row\)=4, but token_column_index=4',
                               lambda: next(iter(at)))

    def test_wrong_text_col(self):
        rows = TestAnnotationTransformer.rows()
        groups = {2: (3,)}
        at = AnnotationTransformer(rows, groups)
        self.assertRaisesRegex(AssertionError,
                               r'column=2 not a list',
                               lambda: next(iter(at)))

    def test_index_error_dropped_cols(self):
        rows = TestAnnotationTransformer.rows()
        groups = TestAnnotationTransformer.groups
        at = AnnotationTransformer(rows, groups, 3, 4)
        self.assertRaisesRegex(RuntimeError,
                               r'row has no column 4 to drop',
                               lambda: next(iter(at)))
