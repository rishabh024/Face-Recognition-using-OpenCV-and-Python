import cv2
import os

#  projectPath variable is the absolute path of the directory where this file resides
projectPath = os.path.dirname(os.path.abspath(__file__))
faceClassifier = cv2.CascadeClassifier(projectPath+'/haarcascade_frontalface_default.xml')

count = 0                  # it is used to count the no of frames
size = 50                  # offset is used for fixed size of images which are captured

name = input('enter your id')
camera = cv2.VideoCapture(0)

while True:
    # camera is used to capture/read the images or frames
    # camera.read() will return the boolean value. If frame is read correctly, then this method will return True, otherwise False.
    retBoolean, frame = camera.read()

    # color of frame is converted to gray color by using COLOR_BGR2GRAY property
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # now, faces are detected by using detectMultiScale() method and stored in the faces variable
    faces = faceClassifier.detectMultiScale(grayFrame, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))




