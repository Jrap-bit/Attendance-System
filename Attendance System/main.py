import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path of Images used for Comparison
path = '/Users/parjanyapandey/Desktop/Programming/PyCharm/College Stuff/Attendance_DIP/Base Models/Face Data'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# Reading the Images and storing their names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    print(classNames)


# Finding the Encodings of each image and storing in a list
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]  # 128 Dimensions for marking specific spots on faces
        encodeList.append(encode)
    return encodeList


# Marking Attendance into Attendance.csv file
def markAttendance(name):
    with open(
            '/Users/parjanyapandey/Desktop/Programming/PyCharm/College Stuff/Attendance_DIP/Base Models/Attendance '
            'System/Output File/Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'n{name},{dtString}\n')


# Storing Encoding of Known Images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Start Video Capture From Webcam
cap = cv2.VideoCapture(0)

while True:
    # Getting Each from of the video one by one
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resizing the image for faster Processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Converting from BGR to RGB

    facesCurFrame = face_recognition.face_locations(imgS)  # Getting location of faces in the image
    encodesCurFrame = face_recognition.face_encodings(imgS,
                                                      facesCurFrame)  # Getting encodings of the faces present in the frame

    # Cycling through the list of faces in the current frame and matching with the known faces.
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        # Resizing the image and adding the name to mark attendance in the csv file
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
