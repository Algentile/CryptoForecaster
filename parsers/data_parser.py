import csv
import numpy as np
from sys import argv
import tensorflow as tf
from tflearn.data_utils import load_csv
import configparser as ConfigParser


def check_feature_num(csv_file, column_number):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        row1 = next(reader)

    if((len(row1) - 1) < column_number or len(row1) == 0):
        raise Exception(
            'Column number out of range for output feature file of column length' + len(row1))


def parse_csv(csv_file, column_number):
    check_feature_num(csv_file, column_number)

    features, labels = load_csv(
        csv_file, target_column=column_number, columns_to_ignore=None, has_header=True)

    feature_tensor = np.array(features).reshape(
        len(features), len(features[0])).astype(np.float)

    label_tensor = np.array(labels).reshape(len(labels), - 1).astype(np.float)

    return feature_tensor, label_tensor


def parse_options(settings_dict, parser, section):
    options = parser.options(section)
    for option in options:
        if(parser.get(section, option) == ''):
            raise Exception(
                'Error when processing config file with option ' + str(option))
        else:
            settings_dict[option] = parser.get(section, option)


def get_config_map(config_file):
    settings_dict = {}
    parser = ConfigParser.ConfigParser()
    parser.read(config_file)
    sections = parser.sections()
    for section in sections:
        parse_options(settings_dict, parser, section)
    return settings_dict
