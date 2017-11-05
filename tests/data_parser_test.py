import unittest
from parsers.data_parser import *


class DataParserTest(unittest.TestCase):

    def setUp(self):
        self.settings_dict = get_config_map('test_files/config.ini')

    def tearDown(self):
        self.settings_dict = None

    def test_parse_config(self):
        train_file = self.settings_dict['train_file']
        self.assertTrue(
            str(train_file) == '/Users/algentile/Documents/CryptoForecaster/datasets/cryptocurrency/eth_train_file.csv')

    def test_broken_config(self):
        with self.assertRaises(Exception) as context:
            self.broken_dict = get_config_map('test_files/broken_config.ini')

        self.assertTrue('Error when processing config file with option column_number' in str(
            context.exception))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(DataParserTest('test_parse_config'))
    suite.addTest(DataParserTest('test_broken_config'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
