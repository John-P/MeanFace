# MeanFace
Create an average (mean) face from a batch on input images (PNG, JPEG, GIF).

For each image a dlib shape predictor is run and the largest detected face is picked. The mean corrdinate of each of the facial landmarks is calculated and then a peicewise affine tranfsorm / warp is applied to align all of the faces. Lastly the mean of each pixel location is calculated.

##Dependencies
- scikit-image
- numpy
- matplotlib
- dlib
