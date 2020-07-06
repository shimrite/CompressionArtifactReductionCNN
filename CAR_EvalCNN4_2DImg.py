import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import time
import sys
import math
from imageio import imread
from imageio import imsave
from itertools import product

tf.compat.v1.disable_eager_execution()


def main(imgs_dir, log_dir, result_dir):

    debug = 0
    save_results = 1
    net1_size = 32
    margin1_pix = 31
    img_sz = 512 # TBD extract from img
    num_of_ptl = np.power(np.divide(img_sz, net1_size), 2)
    patches_to_learn_ind = list(product(list(np.arange(0, img_sz, net1_size)), repeat=2))

    # restore CAR model
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph(os.path.join(log_dir, "model_checkpoint.meta"))
    saver.restore(sess, os.path.join(log_dir, 'model_checkpoint'))
    graph = tf.compat.v1.get_default_graph()
    x = graph.get_tensor_by_name("IteratorGetNext:0")
    prediction = graph.get_tensor_by_name("conv_relu_l_1/Squeeze:0")

    # get all img files names
    jpgImgs = glob.glob(os.path.join(imgs_dir, "*jpg*"))

    # loop on imgs
    num_of_imgs = np.shape(jpgImgs)[0]
    doneImgs = 0
    correctedImgs = np.zeros([num_of_imgs, img_sz, img_sz], dtype=float)
    for i in range(num_of_imgs):
        imgName = jpgImgs[i + doneImgs][-24:-4]
        jImg = imread(jpgImgs[i + doneImgs])

        if np.isnan(np.sum(jImg)):
            sys.stdout.write("\r Slice %d - %s - jpg include NaN. \n" % (i + doneImgs, imgName))
            continue

        if debug:
            plt.figure()
            plt.imshow(jImg)
            plt.title('jpg img')
            plt.show()

        batch_size = 50
        num_of_batches = math.ceil(num_of_ptl / batch_size)
        last_percent = 0
        t1 = time.time()
        for batch_index in range(num_of_batches):
            # init batch patches
            batch_patches = np.zeros([batch_size, net1_size, net1_size], dtype=float)
            batch_patches_ind = np.zeros([batch_size, 4], dtype=int)  # per patch in batch - minx maxx miny maxy
            curr_percent = int(100 * batch_index / num_of_batches)
            if curr_percent > last_percent:
                sys.stdout.write("\rRunning batches: %d%%" % curr_percent)
                sys.stdout.flush()
                last_percent = curr_percent
            for patch_index in range(batch_size):
                global_patch_ind = batch_index * batch_size + patch_index
                if global_patch_ind >= num_of_ptl:
                    batch_patches = batch_patches[:patch_index, :, :]
                    batch_patches_ind = batch_patches_ind[:patch_index, :]
                    break
                min_x = patches_to_learn_ind[global_patch_ind][0]
                max_x = patches_to_learn_ind[global_patch_ind][0] + margin1_pix + 1
                min_y = patches_to_learn_ind[global_patch_ind][1]
                max_y = patches_to_learn_ind[global_patch_ind][1] + margin1_pix + 1
                curr_patch = jImg[min_x:max_x, min_y:max_y]
                curr_patch[curr_patch < 0] = 0
                curr_patch[curr_patch > 255] = 255
                # check valid patch
                if np.isnan(np.sum(curr_patch)):
                    sys.stdout.write("\r Slice %d - %s - jpg include NaN. \n" % (i + doneImgs, imgName))
                    continue

                if not np.shape(curr_patch) == (net1_size, net1_size):
                    continue

                batch_patches[patch_index, :, :] = curr_patch
                batch_patches_ind[patch_index] = [min_x, max_x, min_y, max_y]

            batch_patches_len = batch_patches.shape[0]
            batch_corrected_patches = sess.run(prediction, feed_dict={x: batch_patches}) # [batchSz x 32 x 32] (TBD keep_prob: 1.0)
            for patch_index in range(batch_patches_len):
                curr_patch_ind = batch_patches_ind[patch_index]
                correctedImgs[i, curr_patch_ind[0]:curr_patch_ind[1], curr_patch_ind[2]:curr_patch_ind[3]] = batch_corrected_patches[patch_index]

        if debug:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(jImg)
            plt.title('jpg')
            plt.subplot(1, 2, 2)
            plt.imshow(correctedImgs[i])
            plt.title('corrected')
            plt.show()

        if save_results:
            imsave(result_dir + '/' + imgName + '_car.bmp', correctedImgs[i])

        sys.stdout.write("\r img %d - %s - jpg CAR done. \n" % (i + doneImgs, imgName))

    t2 = time.time()
    print('\n', t2 - t1)
    sys.stdout.write("\r All %d imgs completed! \n" % (i+1))


if __name__=='__main__':

    work_dir = sys.argv[1]                              #'/Users/../CompressionArtifactReduction'
    log_dir = os.path.join(work_dir, sys.argv[2])       #'log'
    imgs_dir = os.path.join(work_dir, sys.argv[3])      #'data')
    result_dir = os.path.join(work_dir, sys.argv[4])    #'results/')

    main(imgs_dir, log_dir, result_dir)