import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from parsers.data_parser import get_user_input

def linear_regression(train_input, train_output, test_input):
	regression = linear_model.LinearRegression()
	regression.fit(train_input, train_output)
	prediction = regression.predict(test_input)
	return prediction

def main():
	file_dict = get_user_input()
	train_input, train_output = file_dict['train']
	test_input, test_output = file_dict['test']
	prediction = linear_regression(train_input, train_output, test_input)
	mean_squared = mean_squared_error(test_output, prediction)
	variance = r2_score(test_output, prediction)
	print('Mean squared is: ', mean_squared)
	print('Variance score is: ', variance)

if __name__ == '__main__':
	main()