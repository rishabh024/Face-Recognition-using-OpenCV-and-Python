import cv2
import os

#  projectPath variable is the absolute path of the directory where this file resides
projectPath = os.path.dirname(os.path.abspath(__file__))
faceClassifier = cv2.CascadeClassifier(projectPath+'/haarcascade_frontalface_default.xml')
