import cv2, os

#  projectPath variable is the absolute path of the directory where this file resides
projectPath = os.path.dirname(os.path.abspath(__file__))

# LBPH (Local Binary Pattern Histogram) algorithm is used to recognize both front and side face.
recognizer = cv2.face.LBPHFaceRecognizer_create()




















