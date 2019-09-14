import cv2
import numpy as np
import deepdish as dd
from matplotlib import pyplot as plt
import seaborn as sns
from threading import Timer, Thread, Event

# cascade_dir = r"C:\Users\Benedetta\Documents\Python_code\stytra_scripts"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# FACE DETECTION ALGORITHM
# Usually some small objects in the background tend to be considered faces by the algorithm,
# so to filter them out we’ll return only the   biggest detected face frame


def detect_faces(img, classifier):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return coords, frame


# EYE DETECTION ALGORITHM
# additional controls: correct position (upper portion) and order (the left first)

def detect_eyes(img, classifier):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(img, 1.3, 5)  # detect eyes

    # the previous function organize in a crescent way the number. With the following "if" I reorganize eyes so
    # that the coords of the left eye are always the first one

    if eyes[0, 0] < eyes[1, 0]:
        c = np.array([eyes[0, :], eyes[1, :]])
    else:
        eyes = np.array([eyes[1, :], eyes[0, :]])

    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None  # even if one eye is not detected, the system will not crush
    right_eye = None  # even if one eye is not detected, the system will not crush
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return eyes, left_eye, right_eye


# cutting eyebrows

def cut_eyebrows(img):
    x = img.shape[0]
    y = img.shape[1]
    height, width = img.shape[:2]
    eyebrow_h = int(height / 3)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    # return x, y, img
    return img


# detecting pupil

def blob_process(image, thresholding_params, blurring_params, detector):
    _, img = cv2.threshold(image, thresholding_params, 255, cv2.THRESH_BINARY)

    # cleaning the puddings
    img[:3, :] = 255
    img[-3:, :] = 255
    img[:, :3] = 255
    img[:, -3:] = 255

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.erode(img, None, iterations=1)

    img = cv2.dilate(img, None, iterations=4)
    img = cv2.blur(img, (blurring_params, blurring_params))

    keypoints = detector.detect(img)

    return keypoints


# creatining a repetitive timer, that will control for the face position every n seconds
from threading import Timer, Thread, Event


class perpetualTimer():

    def __init__(self, t, hFunction):
        self.t = t
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function)

    def handle_function(self):
        self.hFunction()
        self.thread = Timer(self.t, self.handle_function)
        self.thread.start()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()
