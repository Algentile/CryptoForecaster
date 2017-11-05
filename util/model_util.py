import numpy as np 

def next_batch(num, data, labels):
	idx = np.arange(0, len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	shuffled_data = [data[i] for i in idx]
	shuffled_labels = [labels[i] for i in idx]
	return np.array(shuffled_data), np.array(shuffled_labels)