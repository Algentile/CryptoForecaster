import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from parsers.data_parser import *


def linear_regression(train_input, train_output, test_input):
    regression = linear_model.LinearRegression()
    regression.fit(train_input, train_output)
    prediction = regression.predict(test_input)
    return prediction


def main():
    settings_dict = get_config_map('../config.ini')
    
    (train_input, train_output) = parse_csv(settings_dict['train_file'], column_number=3)
    (test_input, test_output) = parse_csv(settings_dict['test_file'], column_number=3)

    prediction = linear_regression(train_input, train_output, test_input)
    mean_squared = mean_squared_error(test_output, prediction)
    variance = r2_score(test_output, prediction)

    print('Mean squared is: ', mean_squared)
    print('Variance score is: ', variance)

if __name__ == '__main__':
    main()
