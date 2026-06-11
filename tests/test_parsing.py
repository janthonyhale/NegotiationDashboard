import csv
import json
import tempfile
import unittest
from pathlib import Path

from services.parsing import allowed, extract_dialogue_language, parse_file


class ParsingTests(unittest.TestCase):
    def write_file(self, name, content):
        path = self.tmpdir / name
        path.write_text(content, encoding='utf-8')
        return path

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_allowed_extensions(self):
        self.assertTrue(allowed('dialogue.JSONL'))
        self.assertFalse(allowed('dialogue.pdf'))
        self.assertFalse(allowed('dialogue'))

    def test_parse_json_messages_and_language(self):
        path = self.write_file(
            'dialogue.json',
            json.dumps({
                'language': 'cn',
                'messages': [{'role': 'Buyer', 'content': 'Need refund', 'irp': 'interest'}],
            }),
        )

        self.assertEqual(extract_dialogue_language(path, 'json'), 'CN')
        self.assertEqual(parse_file(path, 'json'), [{
            'idx': 0,
            'speaker': 'Buyer',
            'text': 'Need refund',
            'ts': None,
            'meta': {'irp_label': 'Interest'},
        }])

    def test_parse_jsonl_skips_metadata_and_reads_language(self):
        path = self.write_file(
            'dialogue.jsonl',
            '\n'.join([
                json.dumps({'language': 'EN'}),
                json.dumps({'speaker': 'Seller', 'text': 'No refund', 'meta': {'irp': 'power'}}),
            ]),
        )

        self.assertEqual(extract_dialogue_language(path, 'jsonl'), 'EN')
        self.assertEqual(parse_file(path, 'jsonl'), [{
            'idx': 0,
            'speaker': 'Seller',
            'text': 'No refund',
            'ts': None,
            'meta': {'irp_label': 'Power'},
        }])

    def test_parse_csv_and_language(self):
        path = self.tmpdir / 'dialogue.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['speaker', 'text', 'irp_label', 'language'])
            writer.writeheader()
            writer.writerow({'speaker': 'Buyer', 'text': 'Please help', 'irp_label': 'right', 'language': 'CN'})

        self.assertEqual(extract_dialogue_language(path, 'csv'), 'CN')
        self.assertEqual(parse_file(path, 'csv'), [{
            'idx': 0,
            'speaker': 'Buyer',
            'text': 'Please help',
            'ts': None,
            'meta': {'irp_label': 'Right'},
        }])

    def test_parse_txt_labels_and_language_header(self):
        language_path = self.write_file('language.txt', 'lang: CN\nBuyer: Hello')
        dialogue_path = self.write_file('dialogue.txt', 'Buyer [Interest]: Hello\nSeller: Hi')

        self.assertEqual(extract_dialogue_language(language_path, 'txt'), 'CN')
        self.assertEqual(parse_file(dialogue_path, 'txt'), [
            {'idx': 0, 'speaker': 'Buyer', 'text': 'Hello', 'ts': None, 'meta': {'irp_label': 'Interest'}},
            {'idx': 1, 'speaker': 'Seller', 'text': 'Hi', 'ts': None, 'meta': {}},
        ])


if __name__ == '__main__':
    unittest.main()
