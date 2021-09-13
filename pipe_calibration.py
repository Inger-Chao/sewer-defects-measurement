# -*- coding: utf-8 -*-
# @Time    : 2021-09-06 14:49
# @Author  : Inger
import cv2
import numpy as np
import os
import os.path as osp
from utils.cv_util import stackImages

def pixDis(a1, b1, a2, b2):
    # distance between points(pixels)
    y = b2 - b1
    x = a2 - a1
    return np.sqrt(x * x + y * y)

def nothing(x):
    # any operation
    pass

font = cv2.FONT_HERSHEY_COMPLEX

global center

def PipeCircle(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    mask = frame.copy()
    canny = cv2.Canny(gray, 100, 50)
    kernel = np.ones((5, 5))
    dil = cv2.dilate(canny, kernel, iterations=1) 
    circles = cv2.HoughCircles(dil,cv2.HOUGH_GRADIENT,2,800,
                            param1=100,param2=50,minRadius=200,maxRadius=800)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(mask, (i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(mask,(i[0],i[1]),2,(0,0,255),3)
    return mask

def ShowVideos(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        mask = PipeCircle(frame)

        imgStack = stackImages(0.3, ([frame, mask]))
        cv2.imshow("Parameters", imgStack)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ShowVideos("/Users/inger/projects/PycharmProjects/opencv_demo/videos/789-779-line.mp4")
def ShowImages(images):
    for image in images:
        while True:
            mask = PipeCircle(image)
            imgStack = stackImages(0.3, ([image, mask]))
            cv2.imshow("Pipe Calibrator", imgStack)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

def ShowDatasetsPipe(path):
    print("Ready to Loading datasets: ",osp.abspath(path))
    images = []
    for file in os.listdir(path):
        filename = osp.join(path, file)
        images.append(cv2.imread(filename))
    print("Imagesets initialized successfully!")
    ShowImages(images)

ShowDatasetsPipe("datasets/JPEGImages")
