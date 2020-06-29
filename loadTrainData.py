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
    return img


def main(imgs_dir, train_tfrecords_path):

    t = time.time()
    # -- Init ---
    net1_size = 32
    margin1_pix = 31
    img_sz = 512
    num_of_ptl = np.power(np.divide(img_sz, net1_size),2)
    patches_to_learn_ind = list(product(list(np.arange(0, img_sz, net1_size)),repeat=2))
    trainNet1_filename_pattern = train_tfrecords_path + '/train_'    # address to save the TFRecords file

    # --- Loop on Imgs ---
    jpg_imgs = glob.glob(os.path.join(imgs_dir, "*jpg*")) # get all img files names
    num_of_imgs = np.shape(jpg_imgs)[0]
    num_of_train_imgs = int(num_of_imgs*0.7) # split data into train-validation sets - TBD! - add shuffle
    train_num_of_files = 0
    train_ptl_global = 0
    done_imgs = 0 # to be updated in case the train tfrecords creation done in two (or more) runs

    for i in range(num_of_train_imgs - done_imgs):
        img_name = jpg_imgs[i + done_imgs][-24:-4]
        jpg_img = imread(jpg_imgs[i])
        bmp_img = imread(jpg_imgs[i][:-3]+"bmp") #not taken using the "glob" method to confirm its the same image only different type
        # input data validation
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
        train_set_size = int(num_of_ptl)
        train_points = patches_to_learn_ind[:train_set_size]

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        train_last_percent = 0
        # -- Loop on Train Points to Learn (per image) --
        for trainPtl_currImg in range(len(train_points)):
            if not trainPtl_currImg % 6000:  # every 60000 points update progress percent
                curr_percent = int(100 * trainPtl_currImg / len(train_points))
                if curr_percent > train_last_percent:
                    sys.stdout.write("\r Bone %d - Writing training tfrecords: %d%%" % (i + done_imgs, curr_percent))
                    sys.stdout.flush()
                    train_last_percent = curr_percent
            if not train_ptl_global % 10000:
                # every 10k points create a TFrecord file
                if train_ptl_global:
                    train_writer_net1.close()
                train_num_of_files = train_num_of_files + 1
                train_net1_filename = trainNet1_filename_pattern + str(train_num_of_files) + ".tfrecords"
                train_writer_net1 = tf.io.TFRecordWriter(train_net1_filename)
            # --- add feature for net ---
            #  Load the current point patches (JPG and BMP)
            img1 = load_patch(train_points[trainPtl_currImg], jpg_img, margin1_pix)
            img2 = load_patch(train_points[trainPtl_currImg], bmp_img, margin1_pix)
            # validate patch loaded correctly
            if not img1.shape == ((margin1_pix + 1), (margin1_pix + 1)):
                sys.stdout.write("\r patch %d - %s - jpg shape fail. \n" % (i + done_imgs, img_name))
                continue
            if not img2.shape == ((margin1_pix + 1), (margin1_pix + 1)):
                sys.stdout.write("\r patch %d - %s - bmp shape fail. \n" % (i + done_imgs, img_name))
                continue

            # Create a feature
            feature1 = {'train/image1': _bytes_feature(tf.compat.as_bytes(img1.tostring())),
                        'train/image2': _bytes_feature(tf.compat.as_bytes(img2.tostring()))}
            # Create an example protocol buffer
            example1 = tf.train.Example(features=tf.train.Features(feature=feature1))
            # Serialize to string and write on the file
            train_writer_net1.write(example1.SerializeToString())
            train_ptl_global = train_ptl_global + 1  # END OF TRAIN POINT

        sys.stdout.write("\r slice %d - %s - Writing training tfrecords Completed.\n" % (i + done_imgs, img_name))
        sys.stdout.write(
            "\r train_num_of_files = %d. total training set = %d ptl.\n" % (train_num_of_files, train_ptl_global))
        sys.stdout.flush()


    # END OF ALL IMAGES POINTS
    train_writer_net1.close()
    elapsed = time.time() - t

    sys.stdout.write("\rWriting Train tfrecords Completed.\n")
    sys.stdout.write("\rTotal elapsed time = %.3f seconds.\n" % elapsed)
    sys.stdout.write("\rTotal training set = %d ptl.\n" % train_ptl_global)
    sys.stdout.flush()


if __name__ == '__main__':

    work_dir = sys.argv[1]  # '/Users/../CompressionArtifactReduction'
    imgs_dir = os.path.join(work_dir, sys.argv[2])  # 'data'
    train_tfrecords_path = os.path.join(work_dir, sys.argv[3])  #'tfrecords/train'

    main(imgs_dir, train_tfrecords_path)
