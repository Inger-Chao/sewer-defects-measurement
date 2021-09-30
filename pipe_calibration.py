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
from thinker.thinker import Defect, Thinker

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
window_caption = 'Pipe Calibrator'
cv2.namedWindow(window_caption)

def Preprocess(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, edge_config.get("canny_threshold1"), edge_config.get("canny_threshold2"))
    return canny

def drawCircle(frame, hough_circles):
    mask = frame.copy()
    for idx, circle in enumerate(hough_circles):
        center = (circle[0],circle[1])
        radius = circle[2]
        cv2.circle(mask, center, radius, (0,255,0), 2)
        cv2.circle(mask, center, 2, (0,0,255), 3)
        if idx == 0:
            cv2.imshow("first calibrated pipe", mask)
    cv2.imshow("optimize circle", mask)

'''
限制缺陷中心点的标定方式
'''
def PipeRadiusRestrictor(frame, defect):
    mindist = max(frame.shape[0], frame.shape[1])
    canny = Preprocess(frame)
    defect_center_x = (defect[1] + defect[3]) / 2
    defect_center_y = (defect[2] + defect[4]) / 2
    distance = pixDis(defect_center_x, defect_center_y, frame.shape[0] / 2, frame.shape[1]/2)
    mask = frame.copy()
    circles = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT, edge_config.get("hough_dp"), 2,
                            param1=edge_config.get("canny_threshold1"),param2=edge_config.get("canny_threshold2"),
                            minRadius=(int)(distance) - 50, maxRadius=int(distance) + 50)
    if circles is None:
        return PipeCircle(frame)
    circles = np.uint16(np.around(circles))
    # reshape [1, i, 3] to [i, 3]
    circles = circles.reshape((circles.shape[1], circles.shape[2]))
    circles = sorted(circles, key=lambda circle:abs(circle[2] - distance))
    drawCircle(frame, circles)
    calibrated_pipe = circles[0] # 取所有行的第 0 列元素
    center = (calibrated_pipe[0],calibrated_pipe[1])
    radius = calibrated_pipe[2]
    pipe = np.zeros((radius * 2, radius * 2),dtype=np.uint8)
    cv2.circle(pipe, (radius, radius), radius, 255, -1)
    cv2.floodFill(pipe, None, (radius, radius), 255, 0,1)
    cv2.circle(mask, center, radius, (0,255,0), 2)
    cv2.circle(mask, center, 2, (0,0,255), 3)
    # cv2.imshow("pipe", pipe)
    return mask, pipe

def PipeCircle(frame):
    mindist = max(frame.shape[0], frame.shape[1])
    mask = frame.copy()
    canny = Preprocess(frame)
    circles = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT, edge_config.get("hough_dp"), 2,
                            param1=edge_config.get("canny_threshold1"),param2=edge_config.get("canny_threshold2"),minRadius=edge_config.get("hough_min_radius"),
                            maxRadius=edge_config.get("hough_max_radius"))
    circles = np.uint16(np.around(circles))
    circles = circles.reshape((circles.shape[1], circles.shape[2]))
    circles = sorted(circles, key=lambda circle:pixDis(circle[0], circle[1], frame.shape[0]/2, frame.shape[1]/2))

    '''
    circles[0] is the calibrated base value.
    return value.(i[0], i[1]) is center, and i[2] is radius.
    the following code for draw the calibrated pipe circle in the copy image.
    '''
    calibrated_pipe = circles[0]
    center = (calibrated_pipe[0],calibrated_pipe[1])
    radius = calibrated_pipe[2]
    pipe = np.zeros((radius * 2, radius * 2),dtype=np.uint8)
    cv2.circle(pipe, (radius, radius), radius, 255, -1)
    cv2.floodFill(pipe, None, (radius, radius), 255, 0,1)
    # circle(img, center, radius, color, thickness=-1)
    cv2.circle(mask, center, radius, (0,255,0), 2)
    cv2.circle(mask, center, 2, (0,0,255), 3)
    # cv2.imshow("pipe", pipe)
    return mask, pipe

def ShowVideos(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        mask, _ = PipeCircle(frame)

        imgStack = stackImages(0.3, ([frame, mask]))
        cv2.imshow(window_caption, imgStack)

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
        defects, features = yolo.detect_image(opencvToPIllow(image))
        result = pil2Opencv(defects)
        detected = []
        if features != None:
            for feat in features:
                mask, pipe = PipeRadiusRestrictor(image, feat)
                dft = Defect(feat[0], image[feat[1]:feat[3], feat[2]:feat[4]])
                detected.append(dft)
            thinker = Thinker(pipe, detected)
            thinker.defect_proportion()
        else:
            ''' Yolo fail to detect defects '''
            mask, pipe = PipeCircle(image)
        imgStack = stackImages(0.3, ([mask, result]))
        cv2.imshow(window_caption, imgStack)
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
