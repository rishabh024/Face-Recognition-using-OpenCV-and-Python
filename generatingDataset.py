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

    # x,y,w,h are 4 cordinates which are used to describe the detected face by drawing the rectangle on the image
    for(x,y,w,h) in faces:
        count += 1

        # now, dataset is formed with the help of captured frames or images
        cv2.imwrite("trainingDataset/face-"+name + "." + str(count) + ".jpg", grayFrame[y-size:y+h+size, x-size:x+w+size])
        cv2.rectangle(frame, (x-size, y-size), (x+w+size, y+h+size), (225, 0, 0), 2)

        # putText() method is used to show the number of images taken for the training dataset
        cv2.putText(frame, 'img-'+str(count), (x, y + h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        # imshow() method is used to show the frames in the window
        cv2.imshow('frame', frame[y-size: y+h+size, x-size: x+w+size])

    # waitKey(1) will wait for key press for one millisecond and it will continue to read frame by using camera.read()
    if(cv2.waitKey(1)==10 or count >= 40):    # if enter key is pressed or if no of images >= 80 then break the loop
        break

camera.release()
cv2.destroyAllWindows()
