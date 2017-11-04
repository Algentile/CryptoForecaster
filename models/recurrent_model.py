import tflearn
import numpy as np
import tensorflow as tf 

class CryptoLSTM:

	def __init__(self, train_input, train_output, test_input, test_output, batch_size=28, rnn_size=128):
		self.train_input = train_input 
		self.train_output = train_output 
		self.test_input = test_input 
		self.test_output = test_output
		self.batch_size = batch_size
		self.rnn_size = rnn_size
		self.n_classes = len(train_input[0])
		self.input = tf.placeholder(tf.float32, len(train_input))
		self.output = tf.placeholder(tf.float32)

	def construct_neural_network(self):
		layer_1 = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.n_classes])),
		'biases': tf.Variable(tf.random_normal([self.n_classes]))}

		self.input = tf.transpose(self.input, [3, 0, 2])
		self.input = tf.reshape(self.input, [-1, self.batch_size])
		self.input = tf.split(0, self.batch_size, self.input)

		print(self.input)

	def train_neural_network():
		pass
