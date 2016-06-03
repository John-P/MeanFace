#!/usr/bin/env python
"""Creates and average face from a list of images"""
from __future__ import print_function
import os
import argparse
import dlib
from skimage import io
from skimage import transform as tr
from skimage import color
import matplotlib.pyplot as plt
import numpy as np

ARG_PARSER = argparse.ArgumentParser(description='Average faces in images.')
ARG_PARSER.add_argument('images', metavar='image', type=str, nargs='+',
                        help='Image files to process')
ARG_PARSER.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Verbose output')
ARG_PARSER.add_argument('-p', '--predictor-path', dest='predictor_path',
                        default="/Users/John/Library/dlib/examples/build/shape_predictor_68_face_landmarks.dat",
                        help="Path to dlib shape predictor")
ARGS = ARG_PARSER.parse_args()

DIR = os.path.dirname(__file__)
MAX_WIDTH = 0
MAX_HEIGHT = 0
FILE_ENDINGS = (".png", ".jpg", ".jpeg", ".gif")

#pylint: disable=no-member
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(ARGS.predictor_path)

def largest_face(face_bounds_list):
    """Pick the largest face from a list of bounding boxes of detected faces"""
    max_face_size = 0
    max_face = None
    for bounds in face_bounds_list:
        face_size = (bounds.left()-bounds.right())**2 + (bounds.top()-bounds.bottom())**2
        if max_face_size < face_size:
            max_face_size = face_size
            max_face = bounds
    return max_face

FACE_LIST = []

for file_name in ARGS.images:
    #Ignore non images files
    if not file_name.lower().endswith(FILE_ENDINGS):
        continue
    if ARGS.verbose:
        print("Processing file: {}".format(file_name))
    file_path = os.path.join(DIR, file_name)
    img = io.imread(file_path)
    #Convert a gray skimage/PIL image to rgb
    if len(img.shape) == 2:
        print("gray2rgb")
        img = color.gray2rgb(img)

    #Detect bounds of faces, second arg means upscale 1 time(s) to help with detection.
    dets = DETECTOR(img, 1)
    if ARGS.verbose:
        print("Number of faces detected: {}".format(len(dets)))
    face = largest_face(dets)
    if face is None:
        continue
    shape = PREDICTOR(img, face)
    if ARGS.verbose:
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

    height, width, _ = img.shape
    MAX_WIDTH = max(MAX_WIDTH, width)
    MAX_HEIGHT = max(MAX_HEIGHT, height)

    FACE_LIST.append((img, face, shape))

LEFT = np.mean([f.left() for _, f, _ in FACE_LIST])
TOP = np.mean([f.top() for _, f, _ in FACE_LIST])
WIDTH = np.mean([f.width() for _, f, _ in FACE_LIST])
HEIGHT = np.mean([f.height() for _, f, _ in FACE_LIST])

SHAPE_POINTS_LISTS = np.array([shape.parts() for _, _, shape in FACE_LIST]).T
MEAN_SHAPE = np.array([(np.mean([p.x for p in point_list]), np.mean([p.y for p in point_list]))
                       for point_list in SHAPE_POINTS_LISTS])

#Move the mean face shape to a good crop position
SM_TFORM = tr.SimilarityTransform(translation=(-LEFT+WIDTH*0.5, -TOP+HEIGHT*0.5))
MEAN_SHAPE = np.array(SM_TFORM(MEAN_SHAPE))
BOTTOM = max([y for _, y in MEAN_SHAPE])
RIGHT = max([x for x, y in MEAN_SHAPE])
CORNERS = np.array([list(xy) for xy in zip([0]*2+[WIDTH*2-1]*2, [HEIGHT*2-1, 0]*2)])
MEAN_SHAPE = np.concatenate([MEAN_SHAPE, CORNERS])

WARPED_IMGS = []

for img, face, shape in FACE_LIST:
    #Affine transform to align with mean shape
    shape_points = np.array([[p.x, p.y] for p in shape.parts()])
    af_tform = tr.estimate_transform('affine', shape_points, MEAN_SHAPE[:-4])
    affine_img = tr.warp(img, af_tform.inverse, output_shape=(int(WIDTH*2), int(HEIGHT*2)))
    shape_points = np.array(af_tform(shape_points))
    #Perform a peicewise affine transform from the face shape to the mean face shape
    #i.e. warp the face image to match the average shape
    pw_af_tform = tr.PiecewiseAffineTransform()
    shape_points = np.concatenate([shape_points, CORNERS])
    pw_af_tform.estimate(shape_points, MEAN_SHAPE)
    warped = tr.warp(affine_img, pw_af_tform.inverse, output_shape=(int(WIDTH*2), int(HEIGHT*2)))

    WARPED_IMGS.append(warped)

#Compute the mean for each pixel, can replace with median etc.
MEAN_IMG = np.mean(WARPED_IMGS, axis=0)

plt.imshow(MEAN_IMG)
plt.show()
