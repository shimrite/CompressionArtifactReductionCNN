from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil
from CAR_DS_Loader import loader
from CAR_ModelCNN import Model

debug = 0
tf.compat.v1.disable_v2_behavior()


def plot_3imgs(img1, img2, img3):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.title('jpg')
    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.title('bmp')
    plt.subplot(1, 3, 3)
    plt.imshow(img3)
    plt.title('pred')
    plt.show()

def main(train_tfrecords_path, val_tfrecords_path, log_dir ):
    # --- Read TFrecordes and create the DataSets ---
    train_TF_dir = [train_tfrecords_path]
    val_TF_dir = [val_tfrecords_path]

    #tf.compat.v1.enable_eager_execution()

    # Initilaze Datasets for training for details see class loader
    # loader return 2 kinds of datasets: train and validation
    train_DS, validation_DS = loader(net_size=net_size, train_batch_size=train_batch_size,
                                         val_batch_size=val_batch_size, train_buffer=train_buffer,
                                            val_buffer=val_buffer).load_from_TFrecordes(train_TF_dir, val_TF_dir)

    # # Create Datasets Iterators (initialized on the step by the string handler)
    train_iterator = train_DS.make_one_shot_iterator()
    val_iterator = validation_DS.make_one_shot_iterator()
    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    iterator = tf.compat.v1.data.Iterator.from_string_handle(handle, train_iterator.output_types)
    next_element = iterator.get_next()

    # --- Build the Model Graph  ---
    # for the deep net details refer to Model Class (modelCAR file)
    model_inputs = next_element
    model = Model(model_inputs)
    prediction = model.predict(model_inputs)
    loss_step = model.calculate_loss(model_inputs, prediction)
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_step) #model.opt_step

    #initiliaze variable
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())

    # --- Start Session ---
    with tf.compat.v1.Session() as sess:
        # initiliaze variables and iterators:
        sess.run(init_op)
        train_iterator_handle = sess.run(train_iterator.string_handle())
        val_iterator_handle = sess.run(val_iterator.string_handle())

        # for tensorboard and saving the weighets
        writer = tf.compat.v1.summary.FileWriter(log_dir, graph=tf.compat.v1.get_default_graph())
        tf.compat.v1.summary.scalar("train_loss", loss_step)
        summary_op = tf.compat.v1.summary.merge_all()
        saver = tf.compat.v1.train.Saver()

        # check if checkpoint exist
        if os.path.exists(log_dir):
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path) # get last train results
                print
                'Variables Restored.'
            else:
                sess.run(tf.compat.v1.global_variables_initializer())
                print
                'Variables Initialized.'
        else:
            sess.run(tf.global_variables_initializer())
            print
            'Variables Initialized.'

        best_val_loss = 100000

        # --- Start Train ---
        for i in range(10000000):
            # train on one batch
            ls, _ = sess.run([loss_step, train_step], feed_dict={handle: train_iterator_handle}) #, feed_dict={keep_prob: dropou}
            if not i % 50:
                # save the nets every 50 iterations. --> net is saved
                saver.save(sess, os.path.join(log_dir, "model_checkpoint"))

            # every 10 train steps run on validation DS
            if not i % 10 and i:
                val_loss_array = np.zeros(10)
                for j in range(10):
                    val_loss = sess.run(loss_step, feed_dict={handle: val_iterator_handle}) #, feed_dict={keep_prob: dropout}
                    val_loss_array[j] = val_loss#[0]
                avg_val_loss = np.average(val_loss_array)
                print('step %d, train loss %g, val loss %g' % (i, ls, avg_val_loss))
                if avg_val_loss <= best_val_loss:
                    saver.save(sess, os.path.join(log_dir, "model_checkpoint"))
                    best_val_loss = avg_val_loss

                debug = 0
                if debug == 1:
                    valInp = sess.run(model_inputs, feed_dict={handle: val_iterator_handle})
                    valPred = sess.run(prediction, feed_dict={model_inputs: valInp}) # Arr - 50 - 32 - 32
                    plot_3imgs(valInp[0][5], valInp[1][5], valPred[5])

                # for tensorboard
                train_sum = tf.compat.v1.summary.Summary(value=[tf.compat.v1.summary.Summary.Value(tag="train_loss", simple_value=ls), ])
                writer.add_summary(train_sum, i)
                val_sum = tf.compat.v1.summary.Summary(value=[tf.compat.v1.summary.Summary.Value(tag="val_loss", simple_value=avg_val_loss), ])
                writer.add_summary(val_sum, i)

            else:
                print('step %d, train loss %g' % (i, ls))


if __name__ == '__main__':

    work_dir = sys.argv[1]  # '/Users/../CompressionArtifactReduction'
    log_dir = os.path.join(work_dir, sys.argv[2])  # 'log'
    train_tfrecords_path = os.path.join(work_dir, sys.argv[3])  #'tfrecords/train'
    val_tfrecords_path = os.path.join(work_dir, sys.argv[4])    #'tfrecords/val'
    # hyper-params default values - TBD (add these params to the argv to enable different config)
    train_batch_size = 1000
    val_batch_size = 1000
    train_buffer = 50
    val_buffer = 50
    dropout = 0.5 # TBD not in use
    net_size = 32

    if os.path.exists(log_dir):
        print
        'log exist.'
    else:
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

    main(train_tfrecords_path, val_tfrecords_path, log_dir )
