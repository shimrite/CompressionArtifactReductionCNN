import glob
import os
import numpy as np
import random
import tensorflow as tf
import sys
import time
import matplotlib.pyplot as plt
from natsort import natsorted, ns
from scipy import stats

plot_Flag = True
DEBUG = False


def _parse_function(path, net_size):
    #this function reading from TFrecords and creating images and labels
    #feature_train is a map function describe how we saved the data in the TFs
    feature_train = {'train/image1': tf.FixedLenFeature([], tf.string),
                     'train/image2': tf.FixedLenFeature([], tf.string)}
    features_train = tf.parse_single_example(path, features=feature_train)
    image1_train = tf.decode_raw(features_train['train/image1'], tf.int16)
    image1_train = tf.cast(image1_train, dtype=tf.float32)
    image1_train = tf.reshape(image1_train, [net_size, net_size])
    image2_train = tf.decode_raw(features_train['train/image2'], tf.int16)
    image2_train = tf.cast(image2_train, dtype=tf.float32)
    image2_train = tf.reshape(image2_train, [net_size, net_size])

    #create standirze all data will be with mean = 0, variance = 1
    #image1_train = tf.image.per_image_standardization(image1_train)

    return image1_train, image2_train


class loader():
    def __init__(self, net_size, train_batch_size, val_batch_size, train_buffer, val_buffer):
        self.net_size = net_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.bone = []
        self.train_buffer = train_buffer
        self.val_buffer = val_buffer



    def _create_DS_TF(self, Files, Train_Flag):
        # filenames = tf.placeholder(tf.string, shape=[None])
        DS = tf.data.TFRecordDataset(Files)
        DS = DS.map(lambda x: _parse_function(x, self.net_size))  #

        if Train_Flag:
            return DS.batch(batch_size=self.train_batch_size).shuffle(buffer_size=self.train_buffer).repeat()
        else:
            return DS.batch(batch_size=self.val_batch_size).shuffle(buffer_size=self.val_buffer).repeat()


    def load_from_TFrecordes(self, source_path, val_source_path):
        # create two sets of data for train and for validation
        trainFiles = glob.glob(os.path.join(source_path[0], "*train_*"))
        if DEBUG:
            trainFiles = glob.glob(os.path.join(source_path, "*train_1*"))

        valFiles = glob.glob(os.path.join(val_source_path[0], "*val_*"))

        train_DS = self._create_DS_TF(trainFiles, True)
        validation_DS = self._create_DS_TF(valFiles, False)

        return train_DS, validation_DS

