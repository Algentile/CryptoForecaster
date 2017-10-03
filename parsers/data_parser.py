import numpy as np
from tflearn.data_utils import load_csv

def parse_csv(csv_file):
	features, labels = load_csv(csv_file, target_column=4, columns_to_ignore=None, has_header=True)
	feature_tensor = np.array(features).reshape(len(features[0]), len(features))
	label_tensor = np.array(labels).reshape(len(labels), 1)
	return feature_tensor, label_tensor