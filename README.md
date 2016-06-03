# MeanFace
Create an average (mean) face from a batch on input images (PNG, JPEG, GIF).

For each image a dlib shape predictor is run and the largest detected face is picked. The mean corrdinate of each of the facial landmarks is calculated and then a peicewise affine tranfsorm / warp is applied to align all of the faces. Lastly the mean of each pixel location is calculated.

##Dependencies
- scikit-image
- numpy
- matplotlib
- dlib

##Usage

    usage: MeanFace.py [-h] [-v] [-p PREDICTOR_PATH] image [image ...]
    
    Average faces in images.
    
    positional arguments:
      image                 Image files to process

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         Verbose output
      -p PREDICTOR_PATH, --predictor-path PREDICTOR_PATH
                            Path to dlib shape predictor
