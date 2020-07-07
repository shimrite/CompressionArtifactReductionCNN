# Future Direction

## Why
As can be seen in the figures below the "patch-wise" learning introduces artifatcs on the patch border:

When looking at the infered image it is even clearer:

These artifacts happens as a result of the 2d Convolution using padding="SAME". (padding of zeros results with higher gray level).
In addition, the loss of the model didnt reached the expected minimum and can be improved.

## How

#### Handling the borders effects:
  1. Overlapped inference with padding="SAME" - borders pixels will be ignored on the inffered image.
  2. Overlapped inference with padding="VALID" - the resulted patch will be smaller than the input patch,hence th eplanning of the patches to be inffered should be adjusted.
  3. Post processing smoothing

#### Model fine-tuning:
1. Batch Normalization (currently there is data standartization per patch).
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
6. train on larger amount of data
  
