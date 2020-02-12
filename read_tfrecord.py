import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

net1_size = 32

data_path = [("C:\\Users\\shimr\\Documents\\work\\testViz\\tfrecords\\train\\train_%d.tfrecords" % (i+1)) for i in range(2)]
with tf.Session() as sess:
    feature = {'train/image1': tf.FixedLenFeature([], tf.string),
               'train/image2': tf.FixedLenFeature([], tf.string)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(data_path, shuffle=True)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image1 = tf.decode_raw(features['train/image1'], tf.int16)
    image2 = tf.decode_raw(features['train/image2'], tf.int16)

    # Reshape image data into the original shape
    image1 = tf.reshape(image1, [net1_size, net1_size])
    image2 = tf.reshape(image2, [net1_size, net1_size])

    # Any preprocessing here ...

    # Creates batches by randomly shuffling tensors
    images1, images2 = tf.train.batch([image1, image2], batch_size=100, capacity=300, num_threads=4, allow_smaller_final_batch=True)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(500):
        img1, img2 = sess.run([images1, images2])
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img1[1,:,:])
        plt.subplot(1, 2, 2)
        plt.imshow(img2[1,:,:])
        plt.title('jpg vs bmp - patch')
        plt.show()
    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()


pass