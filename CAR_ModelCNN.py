import tensorflow as tf
from tensorflow import Tensor
import numpy as np

tf.compat.v1.disable_eager_execution()


class Inputs(object):
    def __init__(self, img1: Tensor, img2: Tensor):
        self.img1 = img1
        self.img2 = img2


class Model(object):
    def __init__(self, inputs: Inputs):
        self.inputs = inputs
        self.predictions = self.predict(inputs)
        self.loss = self.calculate_loss(inputs, self.predictions)
        self.opt_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self.getInp = self.get_inputs(inputs)

    def predict(self, inputs: Inputs):
        with tf.name_scope("image_init"):
            x = inputs[0]
            x = tf.expand_dims(x, -1)
        with tf.name_scope('conv_relu_l'):
            # First convolutional layer - maps one grayscale image to 64 feature maps.
            W_conv1 = self._weight_variable([9, 9, 1, 64])
            b_conv1 = self._bias_variable([64])
            h_conv1 = tf.nn.relu(self._conv2d(x, W_conv1) + b_conv1)
            tf.summary.histogram("layer_1_W", W_conv1)
            tf.summary.histogram("layer_1_B", b_conv1)

            # Second convolutional layer -- maps 64 feature maps to 32.
            W_conv2 = self._weight_variable([7, 7, 64, 32])
            b_conv2 = self._bias_variable([32])
            h_conv2 = tf.nn.relu(self._conv2d(h_conv1, W_conv2) + b_conv2)
            tf.summary.histogram("layer_2_W", W_conv2)
            tf.summary.histogram("layer_2_B", b_conv2)

            # Third convolutional layer -- maps 32 feature maps to 16.
            W_conv3 = self._weight_variable([1, 1, 32, 16])
            b_conv3 = self._bias_variable([16])
            h_conv3 = tf.nn.relu(self._conv2d(h_conv2, W_conv3) + b_conv3)
            tf.summary.histogram("layer_3_W", W_conv3)
            tf.summary.histogram("layer_3_B", b_conv3)

            # Forth convolutional layer -- maps 16 feature maps to 1.
            W_conv4 = self._weight_variable([5, 5, 16, 1])
            b_conv4 = self._bias_variable([1])
            h_conv4 = tf.nn.relu(self._conv2d(h_conv3, W_conv4) + b_conv4)
            h_conv4 = tf.squeeze(h_conv4, axis=3)
            tf.summary.histogram("layer_4_W", W_conv4)
            tf.summary.histogram("layer_4_B", b_conv4)

        return h_conv4

    def calculate_loss(self, inputs: Inputs, h_conv4: Tensor):
        with tf.name_scope('loss'):
            # MSE Loss Function
            loss = tf.reduce_mean(tf.square(h_conv4 - inputs[1]))
            # Add Regularization Loss
            # w_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
            # w_regularizer = tf.keras.regularizers.l2(l=0.01)
            # weights_var = tf.compat.v1.get_variable(name='weights_var', regularizer=w_regularizer)
            # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'weights_var')
            # reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # reg_term = tf.contrib.layers.apply_regularization(w_regularizer, reg_variables)
            # loss += reg_term

        return loss

    def get_inputs(self, inputs: Inputs):
        return inputs

    def _conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _weight_variable(self, shape):
        """weight_variable generates a weight variable of a given shape."""
        # initial = tf.compat.v1.truncated_normal(shape, stddev=0.1)
        req_std = 2
        len_shape = len(shape)
        n = 1
        for i in range(len_shape):
            n = n * shape[i]
        req_std /= n
        req_std = np.sqrt(req_std)
        initial = tf.compat.v1.truncated_normal(shape, stddev=req_std)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
