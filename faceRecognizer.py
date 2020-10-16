import cv2, os

#  projectPath variable is the absolute path of the directory where this file resides
projectPath = os.path.dirname(os.path.abspath(__file__))

# LBPH (Local Binary Pattern Histogram) algorithm is used to recognize both front and side face.
recognizer = cv2.face.LBPHFaceRecognizer_create()

# now, trained model is used to recognise the face of person
recognizer.read(projectPath + '/model.yml')
faceClassifier = cv2.CascadeClassifier(projectPath + '/haarcascade_frontalface_default.xml')


















