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

        # read the image from the dataset and convert it to the gray-scale image using mode "L" and return the converted copy of this image
        grayScaleImageUsingPIL = Image.open(image_path).convert('L')

        # gray-scale image is converted into the numpy array
        image = np.array(grayScaleImageUsingPIL, 'uint8')

        # now, the label of the image is extracted
        label = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
        print('Label-' + str(label))

        # now, faces are detected by using detectMultiScale() method
        faces = faceClassifier.detectMultiScale(image)

        # if the face is detected, append the face to images and append the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(label)










