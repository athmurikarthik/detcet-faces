# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from urllib.request import urlopen
import requests

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

args = {}
args["shape_predictor"] = "shape_predictor_68_face_landmarks.dat"
args["image"] = "images/example_01.jpg"

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


count = 0
f = open("imagefile-test.txt", "r")
for imagePath in f:
    count = count + 1
#     if count <= 12:
#         continue
    print('Image Path ',imagePath )
    response = urlopen(imagePath)
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)

    image = cv2.imdecode(arr, -1)

    # load the input image, resize it, and convert it to grayscale
    # image = cv2.imread(args["image"])
    image = imutils.resize(image, width=500)
    cv2.imshow("actual image", image)
    gray = image if (len(image.shape) == 2) else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        print(rect)
        # rect = dlib.rectangle(10,20,50,100)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        dimensions = face_utils.rect_to_bb(rect)

        x, y, w, h = dimensions[0], dimensions[1], dimensions[2], dimensions[3]

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # loop over the face parts individually

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            # clone = image.copy()
            # cv2.putText(image, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7, (0, 0, 255), 2)

            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            # extract the ROI of the face region as a separate image
            # (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            # roi = image[y:y + h, x:x + w]
            # roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

            # show the particular face part
            # cv2.imshow("ROI", roi)
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)

        # visualize all facial landmarks with a transparent overlay
        # output = face_utils.visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
