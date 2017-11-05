import csv
import numpy as np
from sys import argv
import tensorflow as tf
from tflearn.data_utils import load_csv


def check_number_features(csv_file, column_number):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        row1 = next(reader)
    if((len(row1) - 1) < column_number or len(row1) == 0):
        raise Exception(
            'Column number out of range for output feature file of column length' + len(row1))


def parse_csv(csv_file, column_number):
    check_number_features(csv_file, column_number)
    features, labels = load_csv(csv_file, target_column=column_number, columns_to_ignore=None, has_header=True)
    feature_tensor = np.array(features).reshape(len(features), len(features[0])).astype(np.float)
    label_tensor = np.array(labels).reshape(len(labels), - 1).astype(np.float)

    return feature_tensor, label_tensor

"""
Grabs user input from the command line and returns a dictionary of both the 
training features and labels as well as the testing features and labels.
Input:
	1. Training file --> 70% -- 80% base dataset
	2. testing file  --> 30% -- 20% base dataset
Output:
	file_dict --> dictionary of two tuples containing training and testing tensors.
"""


def get_user_input(col):

    train_file = argv[1]
    test_file = argv[2]

    train_x, train_y = parse_csv(
        '../datasets/cryptocurrencypricehistory/' + train_file, column_number=col)
    test_x, test_y = parse_csv(
        '../datasets/cryptocurrencypricehistory/' + test_file, column_number=col)

    file_dict = {'train': (train_x, train_y), 'test': (test_x, test_y)}

    return file_dict
