# Future Directions

## Why
As can be seen in the figures below the "patch-wise" learning introduces artifatcs on the patch borders:
![Alt text](Fig03_step1000_loss0019.png?raw=true "Title")	
![Alt text](Fig05_step1000_loss0019.png?raw=true "Title")	
When looking at the infered image it is even clearer:
![Alt text](Figure_10_imgCorrected.png?raw=true "Title")	
and if we will zoom-in:
![Alt text](Figure_10_imageCorrected_zoomin.png?raw=true "Title")	

These artifacts happens as a result of the 2d Convolution using padding="SAME". (padding of zeros results with higher gray level).
In addition, the loss of the model didnt reached the expected minimum and can be improved.

## How

#### Handling the borders effects:
  1. Overlapped inference with padding="SAME" - borders pixels will be ignored on the inffered image.
  2. Overlapped inference with padding="VALID" - the resulted patch will be smaller than the input patch on a "valid" padding.
     * on both options above the planning of the patches to be inffered should be adjusted.
  3. Post processing smoothing

#### Model fine-tuning:
1. Batch Normalization (data standartization was tested per patch, didnt recieved better loss and harmed the inference image resulted).
2. Weight Regularization
3. Dropout
4. Model architecture changes
5. Retrain on different hyperparameters sets:

      1. patch_size
      2. batch_size
      3. train-test-validation split
      4. learning_rate
      5. keep-probs
      6. regularization factor

6. Train on larger amount of data
  
