import unittest
import numpy as np
from util.model_util import next_batch
from parsers.data_parser import parse_csv, get_config_map


class ModelUtilTest(unittest.TestCase):

    def setUp(self):
        self.settings_map = get_config_map('test_files/config.ini')

        (self.train_input, self.train_output) = parse_csv(
            self.settings_map['train_file'], column_number=3)
        (self.test_input, self.test_output) = parse_csv(
            self.settings_map['test_file'], column_number=3)

    def tearDown(self):
        self.settings_map = None
        (self.train_input, self.train_output) = None, None
        (self.test_input, self.test_output) = None, None

    def test_next_batch(self):
        (shuffled_input, shuffled_labels) = next_batch(
            int(self.settings_map['batch_size']), self.train_input, self.train_output)

        self.assertTrue(shuffled_input.shape == (28, 5))
        self.assertTrue(shuffled_labels.shape == (28, 1))

def suite():
	suite = unittest.TestSuite()
	suite.addTest(ModelUtilTest('test_next_batch'))
	return suite

if __name__ == '__main__':
	runner = unittest.TextTestRunner()
	runner.run(suite())