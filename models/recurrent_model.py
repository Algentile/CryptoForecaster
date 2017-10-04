import tflearn
import numpy as np
import tensorflow as tf 

class CryptoLSTM:

	def __init__(self, step_num, batch_size, data):
		self.step_num = step_num
		self.batch_size = batch_size
		self.num_features = num_features
		self.input = tf.placeholder(tf.float32, [batch_size, num_steps])
		self.target = tf.placeholder(tf.float32, [batch_size, num_steps])

	def construct_neural_network():

