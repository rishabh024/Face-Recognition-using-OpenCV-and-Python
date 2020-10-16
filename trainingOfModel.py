import cv2,os
import numpy as np
from PIL import Image

# PIL (python image library) is used here for image processing
# projectPath variable is absolute path of the directory where this file resides
projectPath = os.path.dirname(os.path.abspath(__file__))

# LBPH (Local Binary Pattern Histogram) algorithm is used to recognize the faces
recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath = projectPath + '/haarcascade_frontalface_default.xml'

faceClassifier = cv2.CascadeClassifier(cascadePath)
dataPath = projectPath + '/trainingDataset'

def getFacesAndLabels(datapath):

    imagePaths = [os.path.join(datapath, f) for f in os.listdir(datapath)]

    images = []  # images will contain the face images
    labels = []  # labels will contains the label that is assigned to the image

    for image_path in imagePaths:

        # read the image from the dataset and convert it to the grayscale image using mode "L" and return the converted copy of this image
        grayScaleImageUsingPIL = Image.open(image_path).convert('L')



















