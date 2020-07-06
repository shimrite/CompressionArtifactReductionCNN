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
from sklearn.model_selection import train_test_split
from imageio import imsave


# This file load the data, JPG and BMP images, from the
# Split the data into train-validation-test sets (70%-20%-10%) - resulted split is saved under DataSplitList.txt file
# Save the train and validation images (as patches) into TFRecords file (will be used by the DataSet of the model)
# Save the test images in separate folder (will be used by the evaluation file CAR_EvalCNN_2Dimg.py)
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


def is_valid_image(img1, img2, i, done_imgs, img_name):
    # input data validation
    ret_val = 1
    if np.isnan(np.sum(img1)):
        sys.stdout.write("\r Slice %d - %s - jpg include NaN. \n" % (i + done_imgs, img_name))
        ret_val = 0
    if np.isnan(np.sum(img2)):
        sys.stdout.write("\r Slice %d - %s - bmp include NaN. \n" % (i + done_imgs, img_name))
        ret_val = 0
    return ret_val


def plot_2imgs(img1, img2, name1, name2):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(name1+' vs '+name2)
    plt.show()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_patches_from_images(jpg_img, bmp_img, point, margin1_pix, i, done_imgs, img_name):
    #  Load the current point patches (JPG and BMP)
    img1 = load_patch(point, jpg_img, margin1_pix)
    img2 = load_patch(point, bmp_img, margin1_pix)
    # validate patch loaded correctly
    is_valid_patch = 1
    if not img1.shape == ((margin1_pix + 1), (margin1_pix + 1)):
        sys.stdout.write("\r patch %d - %s - jpg shape fail. \n" % (i + done_imgs, img_name))
        is_valid_patch = 0
    if not img2.shape == ((margin1_pix + 1), (margin1_pix + 1)):
        sys.stdout.write("\r patch %d - %s - bmp shape fail. \n" % (i + done_imgs, img_name))
        is_valid_patch = 0
    return [img1, img2, is_valid_patch]


def add_ptl_2tfrecord(jpg_img, bmp_img, point, margin1_pix, i, done_imgs, img_name, train_writer_net1):
    # --- add feature for net ---
    [patch1, patch2, is_valid_patch] = load_patches_from_images(jpg_img, bmp_img, point, margin1_pix, i, done_imgs, img_name)
    if not is_valid_patch:
        return 0
    # Create a feature
    feature1 = {'train/image1': _bytes_feature(tf.compat.as_bytes(patch1.tostring())),
                'train/image2': _bytes_feature(tf.compat.as_bytes(patch2.tostring()))}
    # Create an example protocol buffer
    example1 = tf.train.Example(features=tf.train.Features(feature=feature1))
    # Serialize to string and write on the file
    train_writer_net1.write(example1.SerializeToString())
    return


def save_image_ptls_2tfr(jpg_img, bmp_img, img_name, img_type, patches_to_learn_ind, margin1_pix, num_of_ptl_in_img, train_writer_net1, train_num_of_files, train_tfr_filename_pattern, i, done_imgs, train_ptl_global):
    # -- Loop on Train Points to Learn (per image) --
    random.shuffle(patches_to_learn_ind)
    train_points = patches_to_learn_ind[:num_of_ptl_in_img] # for now train point holds ALL points in image - TBD - in case we want to add data split in image num_of_ptl_in_img will be multiplied in percenatge
    train_last_percent = 0
    for trainPtl_currImg in range(len(train_points)):
        if not trainPtl_currImg % 6000:  # every 60000 points update progress percent
            curr_percent = int(100 * trainPtl_currImg / len(train_points))
            if curr_percent > train_last_percent:
                sys.stdout.write("\r Bone %d - Writing %s tfrecords: %d%%" % (i + done_imgs, img_type, curr_percent))
                sys.stdout.flush()
                train_last_percent = curr_percent
        if not train_ptl_global % 10000:
            # every 10k points create a TFrecord file
            if train_ptl_global:
                train_writer_net1.close()
            # train_writer_net1.close()
            train_num_of_files = train_num_of_files + 1
            train_net1_filename = train_tfr_filename_pattern + str(train_num_of_files) + ".tfrecords"
            train_writer_net1 = tf.io.TFRecordWriter(train_net1_filename)
        # --- add feature for net ---
        add_ptl_2tfrecord(jpg_img, bmp_img, train_points[trainPtl_currImg], margin1_pix, i, done_imgs, img_name,
                          train_writer_net1)
        train_ptl_global = train_ptl_global + 1  # END OF TRAIN POINT

    sys.stdout.write("\r slice %d - %s - Writing into %s tfrecords Completed.\n" % (i + done_imgs, img_name, img_type))
    sys.stdout.write(
        "\r %s_num_of_files = %d. total %s set = %d ptl.\n" % (img_type, train_num_of_files, img_type, train_ptl_global))
    sys.stdout.flush()

    return [train_ptl_global, train_num_of_files, train_writer_net1]


def main(imgs_dir, tfrecords_path):

    t = time.time()
    # -- Init ---
    net1_size = 32
    margin1_pix = 31
    img_sz = 512
    num_of_ptl_in_img = int(np.power(np.divide(img_sz, net1_size),2))
    patches_to_learn_ind = list(product(list(np.arange(0, img_sz, net1_size)),repeat=2))
    train_tfr_filename_pattern = tfrecords_path + '/train/train_'    # address to save the TRAIN TFRecords file
    val_tfr_filename_pattern = tfrecords_path + '/val/val_'  # address to save the VAL TFRecords file
    test_tfr_filename_pattern = tfrecords_path + '/test/test_'  # address to save the TEST TFRecords file
    test_tfr_foldername = tfrecords_path + '/test'  # address to save the TEST TFRecords file

    # --- split data into train-validation-test sets ---
    jpg_imgs = glob.glob(os.path.join(imgs_dir, "*jpg*")) # get all JPG img files names
    bmp_imgs = glob.glob(os.path.join(imgs_dir, "*bmp*")) # get all BMP img files names
    num_of_imgs = np.shape(jpg_imgs)[0]
    x_train0, x_test, y_train0, y_test = train_test_split(jpg_imgs, bmp_imgs, test_size=0.1, random_state=48)
    x_train, x_val, y_train, y_val = train_test_split(x_train0, y_train0, test_size=0.2, random_state=48)
    # Open file "DataSplitList.txt"
    split_list_path = tfrecords_path + '/DataSplitList.txt'
    split_list_file = open(split_list_path, "a")
    split_list_file.write("Following is the data split list: \n")
    # --- Loop on Train Images ---
    sys.stdout.write("\r---- Writing Train tfrecords Starts. -----\n")
    num_of_train_imgs = np.shape(x_train)[0]
    train_num_of_files = 0
    train_ptl_global = 0
    done_imgs = 0 # to be updated in case the train tfrecords creation done in two (or more) runs
    train_net1_filename = train_tfr_filename_pattern + str(train_num_of_files+1)+ ".tfrecords"
    train_writer_net1 = tf.io.TFRecordWriter(train_net1_filename) # this writer wont be used
    split_list_file.write("--- Train Images ---- \n")
    for i in range(num_of_train_imgs - done_imgs):
        img_name = x_train[i + done_imgs][-24:-4]
        split_list_file.write(img_name+"\n")
        jpg_img = imread(x_train[i])
        bmp_img = imread(x_train[i][:-3]+"bmp") #not taken using the y_train to confirm its the same image only different type
        # input data validation
        if not is_valid_image(jpg_img, bmp_img, i, done_imgs, img_name):
            continue
        if debug:
            plot_2imgs(jpg_img, bmp_img, 'jpg', 'bmp')

        # -- Save all Train Points to Learn (per image) --
        [train_ptl_global, train_num_of_files, train_writer_net1] = save_image_ptls_2tfr(jpg_img, bmp_img, img_name, 'train', patches_to_learn_ind, margin1_pix, num_of_ptl_in_img, train_writer_net1, train_num_of_files, train_tfr_filename_pattern, i, done_imgs, train_ptl_global)

    # END OF ALL TRAIN IMAGES POINTS
    train_writer_net1.close()
    elapsed = time.time() - t

    sys.stdout.write("\r---- Writing Train tfrecords Completed. -----\n")
    sys.stdout.write("\rTotal elapsed time = %.3f seconds.\n" % elapsed)
    sys.stdout.write("\rTotal training set = %d ptl.\n" % train_ptl_global)
    sys.stdout.flush()

    # ---- Loop on VAL images ----
    sys.stdout.write("\r---- Writing Validation tfrecords Starts. -----\n")
    t = time.time()
    num_of_val_imgs = np.shape(x_val)[0]
    val_num_of_files = 0
    val_ptl_global = 0
    done_imgs = 0 # to be updated in case the train tfrecords creation done in two (or more) runs
    val_net1_filename = val_tfr_filename_pattern + str(val_num_of_files+1)+ ".tfrecords"
    train_writer_net1 = tf.io.TFRecordWriter(val_net1_filename)  # this writer wont be used
    split_list_file.write("--- Validation Images ---- \n")
    for i in range(num_of_val_imgs - done_imgs):
        img_name = x_val[i + done_imgs][-24:-4]
        split_list_file.write(img_name + "\n")
        jpg_img = imread(x_val[i])
        bmp_img = imread(x_val[i][:-3]+"bmp") #not taken using the y_train to confirm its the same image only different type
        # input data validation
        if not is_valid_image(jpg_img, bmp_img, i, done_imgs, img_name):
            continue
        if debug:
            plot_2imgs(jpg_img, bmp_img, 'jpg', 'bmp')

        # -- Save all Train Points to Learn (per image) --
        [val_ptl_global, val_num_of_files, train_writer_net1] = save_image_ptls_2tfr(jpg_img, bmp_img, img_name, 'val', patches_to_learn_ind, margin1_pix, num_of_ptl_in_img, train_writer_net1, val_num_of_files, val_tfr_filename_pattern, i, done_imgs, val_ptl_global)

    # END OF ALL VAL IMAGES POINTS
    train_writer_net1.close()
    elapsed = time.time() - t

    sys.stdout.write("\r---- Writing Val tfrecords Completed. ----\n")
    sys.stdout.write("\rTotal elapsed time = %.3f seconds.\n" % elapsed)
    sys.stdout.write("\rTotal validation set = %d ptl.\n" % val_ptl_global)
    sys.stdout.flush()

    # ---- Loop on TEST images ----
    sys.stdout.write("\r---- Writing Test images Starts. -----\n")
    t = time.time()
    num_of_test_imgs = np.shape(x_test)[0]
    test_num_of_files = 0
    test_ptl_global = 0
    done_imgs = 0 # to be updated in case the train tfrecords creation done in two (or more) runs
    test_net1_filename = test_tfr_filename_pattern + str(test_num_of_files + 1) + ".tfrecords"
    train_writer_net1 = tf.io.TFRecordWriter(test_net1_filename)  # this writer wont be used
    split_list_file.write("--- Test Images ---- \n")
    for i in range(num_of_test_imgs - done_imgs):
        img_name = x_test[i + done_imgs][-24:-4]
        split_list_file.write(img_name + "\n")
        jpg_img = imread(x_test[i])
        bmp_img = imread(x_val[i][:-3] + "bmp")
        imsave(test_tfr_foldername + '/' + img_name + '.bmp', bmp_img)
        imsave(test_tfr_foldername + '/' + img_name + '.jpg', jpg_img)

    # END OF ALL TEST IMAGES
    elapsed = time.time() - t

    sys.stdout.write("\r---- Writing Test images Completed. ----\n")
    sys.stdout.write("\rTotal elapsed time = %.3f seconds.\n" % elapsed)
    sys.stdout.flush()


    split_list_file.close()

if __name__ == '__main__':

    work_dir = sys.argv[1]  # '/Users/../CompressionArtifactReduction'
    imgs_dir = os.path.join(work_dir, sys.argv[2])  # 'data'
    tfrecords_path = os.path.join(work_dir, sys.argv[3])  #'tfrecords/train'

    main(imgs_dir, tfrecords_path)
