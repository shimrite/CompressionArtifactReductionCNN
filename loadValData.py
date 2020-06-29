import glob
import os
import numpy as np
import random
import tensorflow as tf
import sys
import time
import matplotlib.pyplot as plt
from imageio import imread
from itertools import product

debug = 0


def load_patch(point, img2d, margin_pix):
    min_x = point[0]
    max_x = point[0] + margin_pix +1
    min_y = point[1]
    max_y = point[1] + margin_pix +1
    # crop the image according to the net_size
    img = img2d[min_x:max_x, min_y:max_y]
    # if we cant crop the image - continue
    if img.shape == ((margin_pix + 1), (margin_pix + 1)):
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.int16)
    debug=0
    if debug:
        plt.figure()
        plt.imshow(img)
        plt.show()

    return img


def main(imgs_dir, val_tfrecords_path ):
    t = time.time()
    # Init
    net1_size = 32
    margin1_pix = 31
    img_sz = 512
    num_of_ptl = np.power(np.divide(img_sz, net1_size), 2)
    patches_to_learn_ind = list(product(list(np.arange(0, img_sz, net1_size)), repeat=2))
    val_net1_filename_pattern = os.path.join(val_tfrecords_path, 'val_')  # address to save the TFRecords file

    # --- Loop on Imgs ---
    # get all img files names
    jpg_imgs = glob.glob(os.path.join(imgs_dir, "*jpg*"))
    num_of_imgs = np.shape(jpg_imgs)[0]
    num_of_train_imgs = int(num_of_imgs*0.7)    # TBD - add shuffle
    done_imgs = num_of_train_imgs # the validation set "starts" after the last train test
    val_num_of_files = 0
    val_ptl_global = 0
    for i in range(num_of_imgs - done_imgs):
        # read images
        img_name = jpg_imgs[i + done_imgs][-24:-4]
        jpg_img = imread(jpg_imgs[i])
        bmp_img = imread(jpg_imgs[i][:-3]+"bmp")
        # validation of input data
        if np.isnan(np.sum(jpg_img)):
            sys.stdout.write("\r Slice %d - %s - jpg include NaN. \n" % (i + done_imgs, img_name))
            continue
        if np.isnan(np.sum(bmp_img)):
            sys.stdout.write("\r Slice %d - %s - bmp include NaN. \n" % (i + done_imgs, img_name))
            continue
        if debug:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(jpg_img)
            plt.subplot(1, 2, 2)
            plt.imshow(bmp_img)
            plt.title('jpg vs bmp')
            plt.show()

        random.shuffle(patches_to_learn_ind)
        val_set_size = int(num_of_ptl)
        val_points = patches_to_learn_ind[:val_set_size]

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        val_last_percent = 0
        # -- Loop on Points to Learn (per image) --
        for valPtl_currImg in range(len(val_points)):
            if not valPtl_currImg % 1000:  # every 1000 points update progress percent
                curr_percent = int(100 * valPtl_currImg / len(val_points))
                if curr_percent > val_last_percent:
                    sys.stdout.write("\r slice %d Writing validation tfrecords: %d%%" % (i + done_imgs, curr_percent))
                    sys.stdout.flush()
                    val_last_percent = curr_percent
            if not val_ptl_global % 10000:  # every 10000 points gen new tfrecord file
                if val_ptl_global:
                    val_writer_net1.close()
                val_num_of_files = val_num_of_files + 1
                val1_filename = val_net1_filename_pattern + str(val_num_of_files) + ".tfrecords"
                val_writer_net1 = tf.io.TFRecordWriter(val1_filename)
            # --- add feature for net ---
            #  Load the current point patches (JPG and BMP)
            val_img1 = load_patch(val_points[valPtl_currImg], jpg_img, margin1_pix)
            val_img2 = load_patch(val_points[valPtl_currImg], bmp_img, margin1_pix)
            # validate patch loaded correctly
            if not val_img1.shape == ((margin1_pix + 1), (margin1_pix + 1)):
                sys.stdout.write("\r patch %d - %s - jpg shape fail. \n" % (i + done_imgs, img_name))
                continue
            if not val_img2.shape == ((margin1_pix + 1), (margin1_pix + 1)):
                sys.stdout.write("\r patch %d - %s - bmp shape fail. \n" % (i + done_imgs, img_name))
                continue

            # Create a feature
            val_feature1 = {'train/image1': _bytes_feature(tf.compat.as_bytes(val_img1.tostring())),
                            'train/image2': _bytes_feature(tf.compat.as_bytes(val_img2.tostring()))}
            # Create an example protocol buffer
            val_example1 = tf.train.Example(features=tf.train.Features(feature=val_feature1))
            # Serialize to string and write on the file
            val_writer_net1.write(val_example1.SerializeToString())
            val_ptl_global = val_ptl_global + 1  # END OF VALIDATION POINT
        sys.stdout.write("\r Slice %d - %s - Writing validation tfrecords Completed. \n" % (i + done_imgs, img_name))
        sys.stdout.write("\r val_num_of_files = %d. total validation set = %d ptl.\n" % (val_num_of_files, val_ptl_global))
        sys.stdout.flush()

    # END OF ALL IMAGES POINTS
    val_writer_net1.close()
    elapsed = time.time() - t

    sys.stdout.write("\rWriting Val tfrecords Completed.\n")
    sys.stdout.write("\rTotal elapsed time = %.3f seconds.\n" % elapsed)
    sys.stdout.write("\rTotal training set = %d ptl.\n" % val_ptl_global)
    sys.stdout.flush()


if __name__ == '__main__':

    work_dir = sys.argv[1]  # '/Users/../CompressionArtifactReduction'
    imgs_dir = os.path.join(work_dir, sys.argv[2])  # 'data'
    val_tfrecords_path = os.path.join(work_dir, sys.argv[3])  #'tfrecords/val'

    main(imgs_dir, val_tfrecords_path )


