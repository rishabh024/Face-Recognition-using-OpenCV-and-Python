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

    # now, faces are detected by using detectMultiScale() method
    faces = faceClassifier.detectMultiScale(grayFrame, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        labelPredicted, conf = recognizer.predict(grayFrame[y:y + h, x:x + w])
        cv2.rectangle(frame, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)

        # now, name of the person is displayed on the screen whose face is recognised
        if (labelPredicted == 2):
            labelPredicted = 'Ram'
        elif (labelPredicted == 1):
            labelPredicted = 'Rishabh'
        elif (labelPredicted == 3):
            labelPredicted = 'Dharmesh'








