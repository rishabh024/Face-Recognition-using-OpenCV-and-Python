import cv2, os

#  projectPath variable is the absolute path of the directory where this file resides
projectPath = os.path.dirname(os.path.abspath(__file__))

# LBPH (Local Binary Pattern Histogram) algorithm is used to recognize both front and side face.
recognizer = cv2.face.LBPHFaceRecognizer_create()

# now, trained model is used to recognise the face of person
recognizer.read(projectPath + '/model.yml')
faceClassifier = cv2.CascadeClassifier(projectPath + '/haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)

while True:
    # camera is used to capture the frames for recognition purpose
    retBoolean, frame = camera.read()

    # color of frame is converted to gray color by using COLOR_BGR2GRAY property
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)












