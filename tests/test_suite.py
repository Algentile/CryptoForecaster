from unittest import TestLoader, TextTestRunner, TestSuite
from data_parser_test import DataParserTest
from model_util_test import ModelUtilTest

if __name__ == '__main__':
    loader = TestLoader()

    suite = TestSuite((
        loader.loadTestsFromTestCase(DataParserTest),
        loader.loadTestsFromTestCase(ModelUtilTest)
    ))

    runner = TextTestRunner(verbosity=2)
    runner.run(suite)