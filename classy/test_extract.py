from unittest import TestCase
from classy.extract import row_generator_from_file, Extractor
from tempfile import TemporaryFile


def make_data(rows, columns, text_columns=None, with_quotes=False):
    count = 1
    make_unquoted = lambda c: "This is cell %s." % c
    make_quoted = lambda c: '"This is cell %s."' % c
    make_text = make_quoted if with_quotes else make_unquoted
    data = []

    for r in range(rows):
        cells = list(map(str, range(count, count + columns)))
        count += columns

        if text_columns is not None:
            for col in text_columns:
                cells[col] = make_text(col)

        data.append(cells)

    return data


def make_file(data, encoding='utf-8', sep=','):
    tempfile = TemporaryFile(mode='w+t', encoding=encoding)

    for row in data:
        tempfile.write(sep.join(map(str, row)))
        tempfile.write('\n')

    tempfile.seek(0)
    return tempfile


def remove_quotes(data, columns):
    unquoted_data = []

    for r in range(len(data)):
        row = []
        unquoted_data.append(row)

        for c in range(len(data[r])):
            if c in columns:
                row.append(data[r][c][1:-1])
            else:
                row.append(data[r][c])

    return unquoted_data

class TestRowGeneratorFromFile(TestCase):

    def test_row_generator_from_file(self):
        data = make_data(2, 3)
        file = make_file(data)
        row_gen = row_generator_from_file(file.name)
        count = 0

        for expected, received in zip(data, row_gen):
            self.assertListEqual(expected, received)
            count += 1

        self.assertEqual(count, len(data))

    def test_row_generator_from_file_with_text_columns(self):
        data = make_data(2, 3, text_columns=(1,))
        file = make_file(data)
        row_gen = row_generator_from_file(file.name)
        count = 0

        for expected, received in zip(data, row_gen):
            self.assertListEqual(expected, received)
            count += 1

        self.assertEqual(count, len(data))

    def test_row_generator_from_quoted_file(self):
        data = make_data(2, 3, text_columns=(1,), with_quotes=True)
        file = make_file(data)
        unquoted = remove_quotes(data, (1,))
        row_gen = row_generator_from_file(file.name)
        count = 0

        for expected, received in zip(unquoted, row_gen):
            self.assertListEqual(expected, received)
            count += 1

        self.assertEqual(count, len(data))

    def test_row_generator_with_escapechar(self):
        data = make_data(3, 3, text_columns=(1,), with_quotes=True)
        data[0][1] = '"Cell with comma, here A."'
        data[1][1] = '"Cell with quote "" char."'
        data[2][1] = '"Cell with both "","" chars."'
        file = make_file(data)
        unquoted = remove_quotes(data, (1,))
        row_gen = row_generator_from_file(file.name)
        count = 0
        unquoted[1][1] = 'Cell with quote " char.'
        unquoted[2][1] = 'Cell with both "," chars.'

        for expected, received in zip(unquoted, row_gen):
            self.assertListEqual(expected, received)
            count += 1

        self.assertEqual(count, len(data))

    def test_plain_row_generator_with_escapechar(self):
        data = make_data(2, 3)
        data[0][1] = "cell\\\tA"
        data[1][2] = "cell\tB"
        file = make_file(data, sep='\t')
        row_gen = row_generator_from_file(file.name, dialect='plain')
        count = 0
        data[0][1] = "cell\tA"
        data[1][2] = "cell"
        data[1].append("B")

        for expected, received in zip(data, row_gen):
            self.assertListEqual(expected, received)
            count += 1

        self.assertEqual(count, len(data))


class TestTransformer(TestCase):

    def test_standard_process(self):
        data = make_data(2, 3, text_columns=(1,), with_quotes=True)
        file = make_file(data)
        row_gen = row_generator_from_file(file.name)
        transformer = Extractor(row_gen)
        unquoted = remove_quotes(data, (1,))
        unquoted[0][1] = [['This', 'is', 'cell', '1', '.']]
        unquoted[1][1] = [['This', 'is', 'cell', '1', '.']]
        n = -1

        for n, row in enumerate(transformer):
            self.assertEqual(tuple(unquoted[n]), row)

        self.assertEqual(1, n)

    def _decap_lower_helper(self, decap=False, lower=False):
        data = make_data(2, 3, text_columns=(1,))
        file = make_file(data, sep='\t')
        row_gen = row_generator_from_file(file.name, dialect='plain')
        transformer = Extractor(row_gen, decap=decap, lower=lower)
        unquoted = remove_quotes(data, (1,))
        unquoted[0][1] = [['this', 'is', 'cell', '1', '.']]
        unquoted[1][1] = [['this', 'is', 'cell', '1', '.']]
        n = -1

        for n, row in enumerate(transformer):
            self.assertEqual(tuple(unquoted[n]), row)

        self.assertEqual(1, n)

    def test_decap(self):
        self._decap_lower_helper(decap=True)

    def test_lower(self):
        self._decap_lower_helper(lower=True)

    def test_lower_and_decap(self):
        self._decap_lower_helper(decap=True, lower=True)

    def test_tile_process(self):
        names = ("id", "text", "class")
        data = make_data(2, 3, text_columns=(1,))
        data.insert(0, names)
        file = make_file(data, sep='\t')
        row_gen = row_generator_from_file(file.name, dialect='plain')
        transformer = Extractor(row_gen, has_title=True)
        data[1][1] = [['This', 'is', 'cell', '1', '.']]
        data[2][1] = [['This', 'is', 'cell', '1', '.']]
        self.assertEqual(names, transformer.names)
        n = -1

        for n, row in enumerate(transformer, 1):
            self.assertEqual(tuple(data[n]), row)

        self.assertEqual(2, n)

    def test_sentence_splitting(self):
        data = make_data(1, 3, text_columns=(1,))
        data[0][1] = "This is a sentence. And it is from cell 1."
        file = make_file(data, sep='\t')
        row_gen = row_generator_from_file(file.name, dialect='plain')
        transformer = Extractor(row_gen, decap=True)
        data[0][1] = [['this', 'is', 'a', 'sentence', '.'],
                      ['and', 'it', 'is', 'from', 'cell', '1', '.']]
        n = -1

        for n, row in enumerate(transformer):
            self.assertEqual(tuple(data[n]), row)

        self.assertEqual(0, n)
