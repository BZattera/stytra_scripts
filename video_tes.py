import numpy as np
import cv2
import deepdish as dd
from matplotlib import pyplot as plt
import seaborn as sns
from utilities_eyetracker import *

# cascade_dir = r"C:\Users\Benedetta\Documents\Python_code\stytra_scripts"
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Set up the detector with default parameters.
lower_Area = 300
higher_Area = 1500
# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False
params.filterByArea = True
params.minArea = lower_Area
params.maxArea = higher_Area
detector = cv2.SimpleBlobDetector_create(params)

def nothing(x):
    pass

def video_test():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    counter = 0
    time_interval = 100

    while(True):
        # Capture frame-by-frame
        _, current_frame = cap.read()

        # Our operations on the frame come here
        frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        #if (np.mod(counter, time_interval) == 0):
        print("looking for the face")
        face_frame = detect_face(frame, face_cascade)
        counter + 1
        if face_frame is not None:
            print("face detected")
            eyes = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None:
                    threshold = cv2.getTrackbarPos('threshold', 'image')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Display the resulting frame
        cv2.imshow('image',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


video_test()