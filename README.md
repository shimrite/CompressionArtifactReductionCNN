# CompressionArtifactReductionCNN

## CNN CAR Model for Jpg Images 

The CNN-CAR model uses bmp images as raw data for learning the artifacts added during the jpg compression.
The resulted model reduce the artifacts and save the corrected images.

## The model include the following steps:

1. Divide the data into train-validation-test sets

- 1.1 - Split data to train-validation-test sets (70-20-10)
- 1.2 - Loading the train and validation sets into tfrecords in patches of 32x32.

	The motivation for this patch size starts by the 'Q' matrix of JPG compression algorithm. 
	Since its size of 8x8 I chose to include in the model the "neighbour matrixes", adding local environment to the learn.

- 1.3 - Test set is copied to "test" folder.

2. Train CNN model:
The model is trained on the train images and validated every 10 batches on the val images.
Model Layers:

  - layer 1 - conv&relu, 9x9 kernel, 64 features (map 32x32 image into 32x32x64 features)
  - layer 2 - conv&relu, 7x7 kernel, 32 features (map 32x32x64 features into 32x32x32 features)
  - layer 3 - conv&relu, 1x1 kernel, 16 features (map 32x32x32 features into 32x32x16 features)
  - layer 4 - conv&relu, 5x5 kernel, into 1 feature (results with the original shape image - 32x32)
  
	* loss function (MSE) reduced steadily during the train (both for Train and Validation data)
	* AdamOptimizer with learning_rate=0.001
	* relu activation function
	
3. Run the model on test set - input folder including jpg images - per image, build the batch of patches (each patch 32x32 size), run the net and reconstruct to the image.  


## The package include the following python files:
	1. CAR_LoadData2tfrecords.py - load images and save as patches in TFRecords files 
	2. CAR_ModelCNN.py - the CNN model layers definition
	3. CAR_DS_Loader.py - build the train and validation data sets
	4. CAR_TrainCNN_2Dimg.py - train the net
	5. CAR_EvalCNN4_2DImg.py - evaluate the net on jpg images
	6. read_tfrecord.py - optional, checking the resulted tfrecords
	7. requirementsCAR.txt

## Running the model:
	> CAR_LoadData2tfrecords.py $FileDir$ data tfrecords
	> CAR_TrainCNN_2Dimg.py $FileDir$ log tfrecords/train tfrecords/val
	> CAR_EvalCNN4_2DImg.py $FileDir$ log tfrecords/test results
	
		* $FileDir$ - the working directory
		* data - the folder name that holds the train data set (should be placed under the working dir)
		* tfrecords/train - the folder that will store the train tfrecords (should be placed under the working dir)
		* tfrecords/val - the folder that will store the val tfrecords (should be placed under the working dir)
		* log - the folder that will store the model checkpoint
		* tfrecords/test - the folder name that holds the images to be tested (should be placed under the working dir)
		* results - the folder that will store the corrected images (of the testdata above)
		* pre-requisites: please setup the python environemnt using the requirements file

## Results:
The model was tested on data set of 1000 images [512x512] (each image had two types - JPG and BMP).
After splitting into patches of 32x32 ("points to learn") there are 256,000 patches for train. 

#### --> After ~1000 train steps (train batch size = 1000) we recieved:
	train_loss = 16.892 and val_loss = 17.450 (reduced from ~6000!)
#### --> After ~2000 train steps (train batch size = 1000) we recieved:
	train_loss = 10.292 and val_loss = 11.350!

#### --> Both Train Loss and Validation Loss reduced steadily (no overfitting):
		
![Alt text](Figures/Fig0_train_val_loss.png?raw=true "Title")	

Following is an example of patch before (jpg) and after correction (pred) by the model: 

JPG vs BMP vs JPG after correction - patch size 32x32 - step1000, train_loss=16, val_loss=17
	![Alt text](Figures/Fig1_step1000_loss17.png?raw=true "Title")
JPG vs BMP vs JPG after correction -  patch size 32x32 - step1400, train_loss=14, val_loss=14
	![Alt text](Figures/Fig2_step1480_loss14.png?raw=true "Title")
* Note the artifacts of the JPG compression are removed from the "predicted" patch.

#### --> The model was tested on the test data set --> average test_loss = 10!
	
JPG vs BMP vs JPG after correction - full image 512x512
	
![Alt text](Figures/Figure_10_imgCorrected.png?raw=true "Title")	
	
### * Please refere to the [FutureDirections](FutureDirections.md) file for my next steps for the model improvements.

