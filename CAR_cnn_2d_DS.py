from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import glob
from CAR_DS_Loader import loader
from modelCAR import Model

FLAGS = None
DEBUG = True


train_batch_size = 1000
val_batch_size = 50
net_size = 32
train_buffer = 50
val_buffer = 10
dropout = 0.5

log_dir = 'C:\\Users\\shimr\\Documents\\work\\log'

def main():
    """ reading TFrecordes and creating the datasets"""
    train_TF_dir = ['C:\\Users\\shimr\\Documents\\work\\tfrecords\\train']
    val_TF_dir = ['C:\\Users\\shimr\\Documents\\work\\tfrecords\\val']

    """ initilaze datasets for training for details see class loader
    loader return 2 kinds of datasets: train and validation
    TBD: validate iterator between validation and training sets.
    """
    #generate DS from TFrecords
    train_DS, validation_DS = loader(net_size=net_size, train_batch_size=train_batch_size,
                                         val_batch_size=val_batch_size, train_buffer=train_buffer,
                                            val_buffer=val_buffer).load_from_TFrecordes(train_TF_dir, val_TF_dir)
    #initiliaze iterator, which will feed out net
    iterator = tf.data.Iterator.from_structure(train_DS.output_types, train_DS.output_shapes)
    iterator_val = tf.data.Iterator.from_structure(validation_DS.output_types, validation_DS.output_shapes)
    """next_element object is reading every iteration a new batch.
    next_element[0] = images1, shape=[batch_size, net_size, net_size]
    next_element[1] = images2, shape=[batch size, net_size, net_size]"""
    next_element = iterator.get_next()
    next_element_val = iterator_val.get_next()

    """In order to subtitue between the validation set and the train test create 2 initilazer,
    and everytime we want to move to another set we will initilaze"""
    training_int_op = iterator.make_initializer(train_DS)
    validation_int_op = iterator_val.make_initializer(validation_DS)

    # Build the graph for the deep net, details in model Class
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    model_inputs = next_element
    model = Model(model_inputs)
    prediction = model.predict(model_inputs)
    loss_step = model.calculate_loss(model_inputs, prediction)
    # loss_step = model.loss
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_step) #model.opt_step

    model_inputs_val = next_element_val
    # using the same model for validation DS
    prediction_val = model.predict(model_inputs_val)
    loss_step_val = model.calculate_loss(model_inputs_val, prediction_val)
    train_step_val = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_step_val)

    #initiliaze variable
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        # initiliaze variable and iterator:
        sess.run(training_int_op)
        sess.run(validation_int_op)
        sess.run(init_op)
        #for tensorboard and saving the weighets
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()

        # check if checkpoint exist
        if os.path.exists(log_dir):
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print
                'Variables Restored.'
            else:
                sess.run(tf.global_variables_initializer())
                print
                'Variables Initialized.'
        else:
            sess.run(tf.global_variables_initializer())
            print
            'Variables Initialized.'

        best_val_loss = 100000

        """number of train iteration depend on the amount of the data. for best training its highly recommanded 
        to go throw the data at least 20 times."""
        for i in range(10000000):
            # train on one batch
            ls, _ = sess.run([loss_step, train_step], feed_dict={keep_prob: dropout})
            if not i % 50:
                # for tensorboard
                hist_sum = sess.run(summary_op, feed_dict={keep_prob: dropout})
                writer.add_summary(hist_sum, i)
                #save the nets every 50 iterations.
                saver.save(sess, os.path.join(log_dir, "model_checkpoint"))

            #every 10 train steps run on validation DS
            if not i % 10 and i:
                for j in range(10):
                    val_loss, _ = sess.run([loss_step_val, train_step_val], feed_dict={keep_prob: dropout})

                print('step %d, train loss %g, val loss %g' % (i, ls, val_loss))
                if val_loss <= best_val_loss:
                    saver.save(sess, os.path.join(log_dir, "model_checkpoint"))
                    best_val_loss = val_loss

                debug = 0
                if debug:
                    valInp = sess.run(model_inputs_val)
                    graph = tf.get_default_graph()
                    # x = graph.get_tensor_by_name("IteratorGetNext_1:0")
                    x = graph.get_tensor_by_name("IteratorGetNext:0")
                    # valPred = sess.run(prediction_val, feed_dict={x: valInp[0]}) # Arr - 50 - 32 - 32
                    valPred = sess.run(prediction, feed_dict={x: valInp[0]}) # Arr - 50 - 32 - 32
                    plt.figure()
                    plt.subplot(1, 3, 1)
                    plt.imshow(valInp[0][5])
                    plt.title('jpg')
                    plt.subplot(1, 3, 2)
                    plt.imshow(valInp[1][5])
                    plt.title('bmp')
                    plt.subplot(1, 3, 3)
                    plt.imshow(valPred[5])
                    plt.title('pred')
                    plt.show()

                val_sum = tf.Summary(value=[tf.Summary.Value(tag="val_loss", simple_value=val_loss), ])
                writer.add_summary(val_sum, i)
                # sess.run(training_int_op)

            else:
                print('step %d, train loss %g' % (i, ls))


# if __name__ == '__main__':
if os.path.exists(log_dir):
    print
    'log exist.'
else:
    shutil.rmtree(log_dir)
    os.mkdir(log_dir)
main()
