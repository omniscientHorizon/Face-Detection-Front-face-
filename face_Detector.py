from sre_constants import SUCCESS
import cv2
from random import randrange

# importing trained data
trained_face_data = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# captures video:
webcam = cv2.VideoCapture(0)


# iterate over frames:
while True:

    # read current frame
    successful_frame_read, frame = webcam.read()

    # convert image to greyscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face coordinates
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # make a box
    print(face_coordinates)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256),
                                                  randrange(128, 256), randrange(128, 256)), 2)

    # show face
    cv2.imshow('face detector LFG', frame)
    key = cv2.waitKey(1)

    # to quit process
    if key == 81 or key == 113:
        webcam.release()
        break


print("execution over ")
