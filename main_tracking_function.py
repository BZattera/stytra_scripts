import cv2
import os
import numpy as np
import deepdish as dd
from matplotlib import pyplot as plt
import seaborn as sns
from cv2 import KeyPoint_convert
from utilities_eyetracker_function import *

video = dd.io.load(r"C:\Users\Benedetta\Documents\Python_code\stytra_scripts\190806_f1video.hdf5")
print("this is video shape: ", video.shape )
frame = video[3, :, :]




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



def main():
    face_coord, face_frame = detect_faces(frame, face_cascade)
    # saving coords of the face
    starting_x = 0
    starting_y = 0
    print("these are face coord",face_coord)


    face_x = starting_x + face_coord[0,0]
    face_y = starting_y + face_coord[0,1]

    print("update offsets: ", face_x, face_y)


    if face_frame is not None:
        print("face successfully detected")

        eyes_coord, left_eye, right_eye = detect_eyes(face_frame, eye_cascade)
        print ("this is the left eye : ",left_eye.shape)
        print("this is the right eye : ",right_eye.shape)
        print("this are the eyes shape: ",eyes_coord)
        eyes = np.array([left_eye, right_eye])

        eye_x = [len(eyes)]
        eye_y = [len(eyes)]
        final_x = [len(eyes)]
        final_y = [len(eyes)]
        pt = [len(eyes)]
        eye_track = [len(eyes)]


        for current_eye, eye in enumerate(eyes):
            print("these are eye.shape: ", eye.shape)

            #print("these are eye coord: ", eye_coord[current_eye, 0], eye_coord[current_eye, 1])
            print("these are eye coord: ", eyes_coord[current_eye,0], eyes_coord[current_eye,1])


            eye_x = face_x + eyes_coord[current_eye,0]
            eye_y = face_y + eyes_coord[current_eye,1]

            print("eye_x = ", face_x, " + ",eyes_coord[current_eye,0])
            print("eye_y = ", face_y, " + ", eyes_coord[current_eye,1])


            #eye_x = face_x + eye_coord[current_eye, 0]
            #eye_y = face_y + eye_coord[current_eye, 1]
            print("update offsets: ", eye_x, eye_y)


            if eye is not None:
                threshold = 42
                #print("These are my eye coord: ", eye_coord)
                #cut_x, cut_y, eye = cut_eyebrows(eye)

                new_eye = cut_eyebrows(eye)

                print("this is the new eye: ", new_eye.shape)
                print("to remove x = : ", eyes_coord[current_eye,2], " - ",new_eye.shape[1])
                print("to remove y = : ", eyes_coord[current_eye,3], " - " ,new_eye.shape[0])


                to_remove_y = eyes_coord[current_eye,3] - new_eye.shape[0]

                # adding the dimensions os the 2D transformed eye with the eyebrow cutted
                eye_cut_x = eye_x
                eye_cut_y = eye_y + to_remove_y

                print("eye_cut_y = : ", eye_y, " + ", to_remove_y)


                print("update offsets:", eye_cut_x, eye_cut_y)

                print("eyebrows succesfully cutted")
                keypoints = blob_process(new_eye, threshold, 3, detector)
                #if len(keypoints) > 1:
                 #   keypoints = keypoints[0]
                print("This is my keypoint",keypoints)
                print("key points successfully detected")

                pt = cv2.KeyPoint_convert(keypoints)
                print("this is the keypoint transformed in x y coord: ", pt[0])

                # with the first [0,:] i am sure that, if multiple keypoints are detected, I take only the first one

                kpt_x = int(pt[0,0])
                kpt_y = int(pt[0,1])

                print("this are the useful keypoints: ", kpt_x, kpt_y)




                # adding the dimensions os the 2D transformed eye with the eyebrow cutted

                final_x = eye_cut_x + kpt_x
                final_y = eye_cut_y + kpt_y

                print("final_x = : ", eye_cut_x, " + ", kpt_x)
                print("eye_cut_y = : ", eye_cut_y, " + ", kpt_y)


                print("These are the final eye coords: ", final_x, final_y)

                img = cv2.circle(frame, (final_x, final_y), 5, (255, 0, 0), 5)
                #img = cv2.drawKeypoints(frame, keypoints, img, (255, 0, 0))


    cv2.imshow('image',img)


    cv2.waitKey()






main()