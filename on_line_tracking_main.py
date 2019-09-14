import cv2
import os
import numpy as np
import deepdish as dd
from matplotlib import pyplot as plt
import seaborn as sns
from cv2 import KeyPoint_convert
from utilities_eyetracker import *

# cascade_dir = r"C:\Users\Benedetta\Documents\Python_code\stytra_scripts"
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Set up the detector with default parameters.
lower_Area = 200
higher_Area = 1500
# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
params.filterByCircularity = 1
params.minCircularity = 0.3
params.maxCircularity = 1
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

    while (True):

        # Capture frame-by-frame
        _, current_frame = cap.read()

        # Our operations on the frame come here
        frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if np.mod(counter, time_interval) == 0:
            face_frame, face_frame_gray, left_eye_estimated_position, right_eye_estimated_position, X, Y = detect_face(
                current_frame, frame, face_cascade)

            # OLD: face_coord, face_frame = detect_face(frame, face_cascade)
        # saving coords of the face
        starting_x = 0
        starting_y = 0
        print("these are face coord", X, Y)

        if face_frame is not None:
            face_x = starting_x + X
            face_y = starting_y + Y

            print("update offsets: ", face_x, face_y)

            eyes_coord, left_eye_frame, right_eye_frame, left_eye_frame_gray, right_eye_frame_gray = detect_eyes(
                face_frame,
                face_frame_gray,
                left_eye_estimated_position,
                right_eye_estimated_position,
                eye_cascade)

            if right_eye_frame is not None:
                print("this is the right eye : ", right_eye_frame.shape)
                threshold = cv2.getTrackbarPos('threshold', 'image')

                print("these are eye coord: ", eyes_coord[0, 0], eyes_coord[0, 1])

                right_eye_x = face_x + eyes_coord[0, 0]
                right_eye_y = face_y + eyes_coord[0, 1]

                print("these are the updated offsets: ", right_eye_x, right_eye_y)

                right_keypoints = blob_process(right_eye_frame, threshold, 3, detector)
                right_pt = cv2.KeyPoint_convert(right_keypoints)
                print("this is the right key-point transformed in x y coord: ", right_pt)

                # with the first [0,:] i am sure that, if multiple keypoints are detected, I take only the first one

                right_kpt_x = int(right_pt[0, 0])
                right_kpt_y = int(right_pt[0, 1])

                print("this are the useful key - point: ", right_kpt_x, right_kpt_y)

                right_final_x = right_eye_x + right_kpt_x
                right_final_y = right_eye_y + right_kpt_y

                print("These are the final right eye coords: ", right_final_x, right_final_y)
                draw_blobs('image', right_keypoints)

                #frame = cv2.circle(frame, (right_final_x, right_final_y), 5, (255, 0, 0), 5)

            if left_eye_frame is not None:
                print("this is the left eye : ", left_eye_frame.shape)
                threshold = cv2.getTrackbarPos('threshold', 'image')

                print("these are eye coord: ", eyes_coord[0, 2], eyes_coord[0, 3])

                left_eye_x = face_x + eyes_coord[0, 2]
                left_eye_y = face_y + eyes_coord[0,3]

                print("these are the updated offsets: ", left_eye_x, left_eye_y)

                left_keypoints = blob_process(left_eye_frame, threshold, 3, detector)
                left_pt = cv2.KeyPoint_convert(left_keypoints)
                print("this is the right key-point transformed in x y coord: ", left_pt)

                # with the first [0,:] i am sure that, if multiple keypoints are detected, I take only the first one

                left_kpt_x = int(left_pt[0, 0])
                left_kpt_y = int(left_pt[0, 1])

                print("this are the useful key - point: ", left_kpt_x, left_kpt_y)

                left_final_x = left_eye_x + left_kpt_x
                left_final_y = left_eye_y + left_kpt_y

                print("These are the final right eye coords: ", left_final_x, left_final_y)

                draw_blobs('image', left_keypoints)

                #frame = cv2.circle(frame, (left_final_x, left_final_y), 5, (255, 0, 0), 5)

        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


video_test()

