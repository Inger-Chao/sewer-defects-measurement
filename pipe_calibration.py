# -*- coding: utf-8 -*-
# @Time    : 2021-09-06 14:49
# @Author  : Inger
import cv2
import numpy as np
import os
import os.path as osp
from utils.cv_util import stackImages
from utils.utils import opencvToPIllow, pil2Opencv
from detector.defects_detector import YOLO
from config import edge_config, conf

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
    mask = frame.copy()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, edge_config.get("canny_threshold1"), edge_config.get("canny_threshold2"))
    kernel = np.ones((5, 5))
    dil = cv2.dilate(canny, kernel, iterations=1) 
    cv2.imshow("binary image", dil)
    circles = cv2.HoughCircles(dil,cv2.HOUGH_GRADIENT,2, edge_config.get("hough_min_dist"),
                            param1=255,param2=50,minRadius=edge_config.get("hough_min_radius"),
                            maxRadius=edge_config.get("hough_max_radius"))
    circles = np.uint16(np.around(circles))
    assert len(circles) == 1, 'There are more that one circle detected!'
    '''
    circles[0] is the calibrated base value.
    return value.(i[0], i[1]) is center, and i[2] is radius.
    the following code for draw the calibrated pipe circle in the copy image.
    '''
    circle = circles[0]
        # circle(img, center, radius, color, thickness=-1)
    cv2.circle(mask, (circle[0],circle[1]), circle[2], (0,255,0), 2)
    cv2.circle(mask, (circle[0],circle[1]), 2, (0,0,255), 3)
    return mask, circle

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
    yolo = YOLO()
    for file in images:
        print('==>Processing: ', file)
        image = cv2.imread(file)
        mask, _ = PipeCircle(image)
        defects, _ = yolo.detect_image(opencvToPIllow(image))
        result = pil2Opencv(defects)
        imgStack = stackImages(0.3, ([mask, result]))
        cv2.imshow("Pipe Calibrator", imgStack)
        cv2.waitKey()
    cv2.destroyAllWindows()

def ShowDatasetsPipe(path):
    print("Ready to Loading datasets: ",osp.abspath(path))
    images = []
    for file in os.listdir(path):
        filename = osp.join(path, file)
        images.append(filename)
    print("Imagesets initialized successfully!")
    ShowImages(images)

ShowDatasetsPipe(conf.get("datasets"))
