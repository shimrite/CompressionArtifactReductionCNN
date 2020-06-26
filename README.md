# CompressionArtifactReductionCNN

## CNN CAR model for jpg images 

The CNN-CAR model uses bmp images as raw data for learning the artifacts added during the jpg compression.
The resulted model reduce the artifacts and save the corrected images.

The model include the following steps:

1. loading the train and validation sets into tfrecords
2. train the net
3. run the net on test set

The package include the following python files:

1. loadTrainData 
2. loadValData
3. read_tfrecord (optional, test the resulted tfrecords)
4. modelCAR (include the CNN model layers definition)
5. CAR_DS_Loader (build the train and validation data sets)
6. CAR_cnn_2d_DS (train the net)
7. runCARCNN4Img2d (evaluate the net on jpg images)

In order to run the model, please update the following:

1. loadTrainData - 
	imgs_dir = 'C:\\Users\\..\\trainData' # path to the validation set directory
	valNet1_filename_pattern = 'C:/Users/../tfrecords/train/train_'  # address to save the TFRecords file

2. loadValData - 
	imgs_dir = 'C:\\Users\\..\\valData' # path to the validation set directory
	valNet1_filename_pattern = 'C:/Users/../tfrecords/val/val_'  # address to save the TFRecords file

3. read_tfrecord -
	data_path = [("C:\\Users\\..\\tfrecords\\train\\train_%d.tfrecords" % (i+1)) for i in range(2)]

4. modelCAR - none

5. CAR_DS_Loader - none

6. CAR_cnn_2d_DS - 
	log_dir = 'C:\\Users\\..\\log'
	train_TF_dir = ['C:\\Users\\..\tfrecords\\train']
	val_TF_dir = ['C:\\Users\\..\\tfrecords\\val']

7. runCARCNN4Img2d -
    imgs_dir = 'C:\\Users\\..\\valData'
    log_dir = 'C:\\Users\\..\\log'
	result_dir = 'C:/Users/../results/'
    debug = 0 			# display images while runing
    save_results = 1 	# save corrected images 
