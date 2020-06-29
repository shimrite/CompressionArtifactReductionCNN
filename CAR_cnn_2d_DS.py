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
from modelCAR import Model

debug = 0


def main(train_tfrecords_path, val_tfrecords_path, log_dir ):
    # --- Read TFrecordes and create the DataSets ---
    train_TF_dir = [train_tfrecords_path]
    val_TF_dir = [val_tfrecords_path]

    # Initilaze Datasets for training for details see class loader
    # loader return 2 kinds of datasets: train and validation
    train_DS, validation_DS = loader(net_size=net_size, train_batch_size=train_batch_size,
                                         val_batch_size=val_batch_size, train_buffer=train_buffer,
                                            val_buffer=val_buffer).load_from_TFrecordes(train_TF_dir, val_TF_dir)
    # Create and Initiliaze Datasets Iterators
    iterator = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(train_DS), tf.compat.v1.data.get_output_shapes(train_DS))
    iterator_val = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(validation_DS), tf.compat.v1.data.get_output_shapes(validation_DS))

    # In order to subtitue between the validation set and the train test create 2 initilazers
    training_int_op = iterator.make_initializer(train_DS)
    validation_int_op = iterator_val.make_initializer(validation_DS)

    # next_element object is reading every iteration a new batch.
    # next_element[0] = images1 (JPG), shape=[batch_size, net_size, net_size]
    # next_element[1] = images2 (BMP), shape=[batch size, net_size, net_size]
    next_element = iterator.get_next()
    next_element_val = iterator_val.get_next()

    tf.compat.v1.disable_eager_execution()

    # --- Build the Model Graph  ---
    # for the deep net details refer to Model Class (modelCAR file)
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    model_inputs = next_element
    model = Model(model_inputs)
    prediction = model.predict(model_inputs)
    loss_step = model.calculate_loss(model_inputs, prediction)
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss_step) #model.opt_step
    model_inputs_val = next_element_val
    # using the same model for validation DS
    prediction_val = model.predict(model_inputs_val)
    loss_step_val = model.calculate_loss(model_inputs_val, prediction_val)
    train_step_val = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss_step_val)

    #initiliaze variable
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())

    # --- Start Session ---
    with tf.compat.v1.Session() as sess:
        # initiliaze variables and iterators:
        sess.run(training_int_op)
        sess.run(validation_int_op)
        sess.run(init_op)
        # for tensorboard and saving the weighets
        writer = tf.compat.v1.summary.FileWriter(log_dir, graph=tf.compat.v1.get_default_graph())
        tf.compat.v1.summary.scalar("loss", loss_step_val)
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
            ls, _ = sess.run([loss_step, train_step], feed_dict={keep_prob: dropout})
            if not i % 50:
                # for tensorboard
                hist_sum = sess.run(summary_op, feed_dict={keep_prob: dropout})
                writer.add_summary(hist_sum, i)
                # # save the nets every 50 iterations. --> net is saved
                # saver.save(sess, os.path.join(log_dir, "model_checkpoint"))

            # every 10 train steps run on validation DS
            if not i % 10 and i:
                val_loss_array = np.zeros(10)
                for j in range(10):
                    # val_loss, _ = sess.run([loss_step_val, train_step_val], feed_dict={keep_prob: dropout})
                    val_loss = sess.run([loss_step_val], feed_dict={keep_prob: dropout})
                    val_loss_array[j] = val_loss[0]
                avg_val_loss = np.average(val_loss_array)
                print('step %d, train loss %g, val loss %g' % (i, ls, avg_val_loss))
                if avg_val_loss <= best_val_loss:
                    saver.save(sess, os.path.join(log_dir, "model_checkpoint"))
                    best_val_loss = avg_val_loss

                if debug:
                    valInp = sess.run(model_inputs_val)
                    graph = tf.compat.v1.get_default_graph()
                    x = graph.get_tensor_by_name("IteratorGetNext:0")
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

                # for tensorboard
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
    val_batch_size = 50
    train_buffer = 50
    val_buffer = 10
    dropout = 0.5
    net_size = 32

    if os.path.exists(log_dir):
        print
        'log exist.'
    else:
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

    main(train_tfrecords_path, val_tfrecords_path, log_dir )
