import numpy as np
from sys import argv
from tflearn.data_utils import load_csv

def parse_csv(csv_file):
	features, labels = load_csv(csv_file, target_column=4, columns_to_ignore=None, has_header=True)
	feature_tensor = np.array(features).reshape(len(features[0]), len(features))
	label_tensor = np.array(labels).reshape(len(labels), 1)
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
def get_user_input():
	train_file = argv[1]
	test_file = argv[2]
	train_x, train_y = parse_csv('../datasets/cryptocurrencypricehistory/' + train_file)
	test_x, test_y = parse_csv('../datasets/cryptocurrencypricehistory/' + test_file)
	file_dict = {'train': (train_x, train_y), 'test': (test_x, test_y)}
	return file_dict

