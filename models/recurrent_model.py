import tflearn
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class CryptoLSTM:

    def __init__(self, train_input, train_output, test_input, test_output, batch_size=28, rnn_size=128):
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.n_classes = len(train_input[0])
        self.input_placeholder = tf.placeholder(
            'float', [None, len(train_input), len(train_input[0])])
        self.output_placeholder = tf.placeholder('float')

    def get_train_input(self):
    	return self.train_input

    def set_train_input(self, train_input):
    	self.train_input = train_input 

    def get_train_output(self):
    	return self.train_output

    def set_train_output(self, train_output):
    	self.train_output = train_output 

    def get_test_input(self):
    	return self.test_input

    def set_test_input(self, test_input):
    	self.test_input = test_input

    def get_test_output(self):
    	return self.test_output

    def set_test_output(self, test_output):
    	self.test_output = test_output 

    def get_batch_size(self):
    	return self.batch_size

    def set_batch_size(self, batch_size):
    	self.batch_size = batch_size 

    def get_rnn_size(self):
    	return self.rnn_size

    def set_rnn_size(self, rnn_size):
    	self.rnn_size = rnn_size

    def get_n_classes(self):
    	return self.n_classes

    def set_n_classes(self, n_classes):
    	self.n_classes = n_classes

    def get_input_placeholder(self):
    	return self.input_placeholder

    def set_input_placeholder(self, input_placeholder):
    	self.input = input_placeholder

    def get_output_placeholder(self):
    	return self.output_placeholder

    def set_output_placeholder(self, output_placeholder):
    	self.output_placeholder = output_placeholder

    def construct_neural_network(self):

        layer_1 = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.n_classes])),
                   'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        lstm_cell = rnn.BasicLSTMCell(self.rnn_size)

        self.input_placeholder = tf.unstack(tf.transpose(
            self.input_placeholder, tf.TensorShape([1, 0, 2])))

        outputs, states = rnn.static_rnn(
            lstm_cell, self.input_placeholder, dtype=tf.float32)

        output_layers = tf.matmul(
            outputs[-1], layer_1['weights']) + layer_1['biases']

    def train_neural_network():
    	pass