import glob
import os
import numpy as np
import random
import tensorflow as tf
import sys
import time
import matplotlib.pyplot as plt
from natsort import natsorted, ns
from scipy.misc import imread
from itertools import product

def load_patch(point, img2d, margin_pix):
    min_x = point[0]
    max_x = point[0] + margin_pix +1
    min_y = point[1]
    max_y = point[1] + margin_pix +1
    #min_z = point[2] - margin_pix[1]
    #max_z = point[2] + margin_pix[1] + 1
    # crop the image according to the net_size
    img = img2d[min_x:max_x, min_y:max_y]#, min_z:max_z]
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


t = time.time()

net1_size = 32
margin1_pix = 31
img_sz = 512
num_of_ptl = np.power(np.divide(img_sz, net1_size), 2)
patches_to_learn_xInd = list(np.arange(0, img_sz, net1_size))
patches_to_learn_yInd = list(np.arange(0, img_sz, net1_size))
patches_to_learn_ind = list(product(list(np.arange(0, img_sz, net1_size)), repeat=2))

imgs_dir = 'C:\\Users\\shimr\\Documents\\work\\valData'
log_dir = 'C:\\Users\\shimr\\Documents\\work\\log'

# get all img files names
jpgImgs = glob.glob(os.path.join(imgs_dir, "*jpg*"))
bmpImgs = glob.glob(os.path.join(imgs_dir, "*bmp*"))

# loop on imgs
num_of_imgs = np.shape(jpgImgs)[0]
train_num_of_files = 0
val_num_of_files = 0
train_ptl_tot = 0
val_ptl_tot = 0
trainPtl_global = 0
valPtl_global = 0
doneImgs = 0

for i in range(num_of_imgs - doneImgs):
    seriesNum = jpgImgs[i+doneImgs][-17:-14]
    sliceNum = jpgImgs[i + doneImgs][-7:-4]
    imgName = jpgImgs[i + doneImgs][-24:-4]
    jImg = imread(jpgImgs[i])#, flatten=True, mode='I')
    bImg = imread(jpgImgs[i][:-3]+"bmp")#, flatten=True, mode='I')

    if np.isnan(np.sum(jImg)):
        sys.stdout.write("\r Slice %d - %s - jpg include NaN. \n" % (i + doneImgs, imgName))
        continue

    if np.isnan(np.sum(bImg)):
        sys.stdout.write("\r Slice %d - %s - bmp include NaN. \n" % (i + doneImgs, imgName))
        continue

    debug = 0
    if debug:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(jImg)
        plt.subplot(1, 2, 2)
        plt.imshow(bImg)
        plt.title('jpg vs bmp')
        plt.show()

    log_dir = os.path.join(log_dir, "log_{}".format(margin1_pix))

    jImg2d = jImg #np.reshape(jImg, [img_sz, img_sz], 'F')
    bImg2d = bImg #np.reshape(bImg, [img_sz, img_sz], 'F')

    random.shuffle(patches_to_learn_ind)
    val_set_size = int(num_of_ptl)
    val_points = patches_to_learn_ind[:val_set_size]

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    valNet1_filename_pattern = 'C:/Users/shimr/Documents/work/tfrecords/val/val_'  # address to save the TFRecords file

    val_last_percent = 0
    for valPtl_currImg in range(len(val_points)):
        if not valPtl_currImg % 1000:  # every 1000 points update progress percent
            curr_percent = int(100 * valPtl_currImg / len(val_points))
            if curr_percent > val_last_percent:
                sys.stdout.write("\r slice %d Writing validation tfrecords: %d%%" % (i + doneImgs, curr_percent))
                sys.stdout.flush()
                val_last_percent = curr_percent
        if not valPtl_global % 10000:  # every 10000 points gen new tfrecord file
            if valPtl_global:
                valWriterNet1.close()
            val_num_of_files = val_num_of_files + 1
            val1_filename = valNet1_filename_pattern + str(val_num_of_files) + ".tfrecords"
            valWriterNet1 = tf.python_io.TFRecordWriter(val1_filename)
        # --- add feature for net ---
        # Load the image
        val_img1 = load_patch(val_points[valPtl_currImg], jImg2d, margin1_pix)
        val_img2 = load_patch(val_points[valPtl_currImg], bImg2d, margin1_pix)

        if not val_img1.shape == ((margin1_pix + 1), (margin1_pix + 1)):
            sys.stdout.write("\r patch %d - %s - jpg shape fail. \n" % (i + doneImgs, imgName))
            continue
        if not val_img2.shape == ((margin1_pix + 1), (margin1_pix + 1)):
            sys.stdout.write("\r patch %d - %s - bmp shape fail. \n" % (i + doneImgs, imgName))
            continue

        # Create a feature
        val_feature1 = {'train/image1': _bytes_feature(tf.compat.as_bytes(val_img1.tostring())),
                        'train/image2': _bytes_feature(tf.compat.as_bytes(val_img2.tostring()))}
        # Create an example protocol buffer
        val_example1 = tf.train.Example(features=tf.train.Features(feature=val_feature1))
        # Serialize to string and write on the file
        valWriterNet1.write(val_example1.SerializeToString())
        valPtl_global = valPtl_global + 1  # END OF VALIDATION POINT
    sys.stdout.write("\r Slice %d - %s - Writing validation tfrecords Completed. \n" % (i + doneImgs, imgName))
    sys.stdout.write("\r val_num_of_files = %d. total validation set = %d ptl.\n" % (val_num_of_files, valPtl_global))
    sys.stdout.flush()

# END OF ALL IMAGES POINTS
valWriterNet1.close()
elapsed = time.time() - t

sys.stdout.write("\rWriting tfrecords Completed.\n")
sys.stdout.write("\rTotal elapsed time = %.3f seconds.\n" % elapsed)
sys.stdout.write("\rTotal training set = %d ptl.\n" % valPtl_global)
sys.stdout.flush()


