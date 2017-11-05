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
        self.input = tf.placeholder(
            'float', [None, len(train_input), len(train_input[0])])
        self.output = tf.placeholder('float')

    def construct_neural_network(self):
        layer_1 = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.n_classes])),
                   'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        lstm_cell = rnn.BasicLSTMCell(self.rnn_size)

        self.input = tf.unstack(tf.transpose(
            self.input, tf.TensorShape([1, 0, 2])))

        outputs, states = rnn.static_rnn(
            lstm_cell, self.input, dtype=tf.float32)

        output_layers = tf.matmul(
            outputs[-1], layer_1['weights']) + layer_1['biases']

    def train_neural_network():
    	pass