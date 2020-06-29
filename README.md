# CompressionArtifactReductionCNN

## CNN CAR Model for Jpg Images 

The CNN-CAR model uses bmp images as raw data for learning the artifacts added during the jpg compression.
The resulted model reduce the artifacts and save the corrected images.

## The model include the following steps:

1. Divided the data into train-validation-test sets
	1.1. Loading the train and validation sets into tfrecords in patches of 32x32
	(allowing kernels of 9x9 later in the net, adding local environment to the learn, 'Q' matrix of jpg compression is 8x8..)
	1.2. Train and Validation sets loaded into two Datasets ('map' function reads from the tfrecords).
2. Train CNN model:
  - layer 1 - conv&relu, 9x9 kernel, 64 features (map 32x32 image into 32x32x64 features)
  - layer 2 - conv&relu, 7x7 kernel, 32 features (map 32x32x64 features into 32x32x32 features)
  - layer 3 - conv&relu, 1x1 kernel, 16 features (map 32x32x32 features into 32x32x16 features)
  - layer 4 - conv&relu, 5x5 kernel, into 1 feature (results with the original shape image - 32x32)
  
	* loss function (MSE) reduced steadily during the train
	* AdamOptimizer with learning_rate=0.001
	* relu activation function
3. Run the net on test set - input folder including jpg images - per image, build the batch of patches (each patch 32x32 size), run the net and reconstruct to the image.  


## The package include the following python files:
	1. loadTrainData.py 
	2. loadValData.py
	3. read_tfrecord.py (optional, checking the resulted tfrecords)
	4. modelCAR.py (include the CNN model layers definition)
	5. CAR_DS_Loader.py (build the train and validation data sets)
	6. CAR_cnn_2d_DS.py (train the net)
	7. runCARCNN4Img2d.py (evaluate the net on jpg images)
	8. requirementsCAR.txt

## In order to run the model, please setup the python environemnt using the erquirements file and run the following:
	> loadTrainData.py $FileDir$ data tfrecords/train 
	> loadValData.py $FileDir$ data tfrecords/val 
	> CAR_cnn_2d_DS.py $FileDir$ log tfrecords/train tfrecords/val
	> runCARCNN4Img2d.py $FileDir$ log testdata results
	
		* $FileDir$ - the working directory
		* data - the folder name that holds the train data set (should be placed under the working dir)
		* tfrecords/train - the folder that will store the train tfrecords (should be placed under the working dir)
		* tfrecords/val - the folder that will store the val tfrecords (should be placed under the working dir)
		* log - the folder that will store the model checkpoint
		* testdata - the folder name that holds the train data set (should be placed under the working dir)
		* results - the folder that will store the corrected images (of the testdata above)

## Results:
	The model was tested on data set of 1000 images [512x512] (each image had two types - JPG and BMP).
	After splitting into patches of 32x32 ("points to learn") there are 256,000 patches for train. 
	After ~800 train steps (train batch size = 1000) we recieved:
	train_loss = 1.792 and val_los = 2.880

Following is an example of patch before (jpg) and after correction (pred) by the model: 
	![Alt text](Figure_2.png?raw=true "Title")
	![Alt text](Figure_1.png?raw=true "Title")
* Note the artifacts of the JPG compression are removed from the "predicted" patch.
