# -*- coding: utf-8 -*-
# @Time    : 2021-09-06 14:49
# @Author  : Inger
from typing import Set
import cv2
import numpy as np
import os
import os.path as osp
from utils.cv_util import save_image, stackImages, pixDis
from utils.utils import opencvToPIllow, pil2Opencv, isImageFile
from detector.defects_detector import YOLO
from config import edge_config, conf, dft_rank_tbl
from thinker.thinker import Defect, Thinker


font = cv2.FONT_HERSHEY_COMPLEX

global center
window_caption = 'Pipe Calibrator'
# cv2.namedWindow(window_caption)

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

def ReturnedCircle(mask, hough_circle):
    if len(hough_circle) == 0:
        '''default calibrated pipe is about 1/3 of entire picture'''
        print("[ERROR][APC] Fail to calibrate pipe")
        center = ((int)(mask.shape[1] * 0.5), int(mask.shape[0] * 0.5))
        radius = min((int)(mask.shape[0] * 0.3), (int)(mask.shape[1] * 0.3))
        pipe = np.zeros((radius * 2, radius * 2),dtype=np.uint8)
        cv2.circle(pipe, (radius, radius), radius, 255, -1)
        cv2.circle(mask, center, radius, (0,255,0), 2)
        cv2.circle(mask, center, 2, (0,0,255), 3)
        return mask, pipe
    calibrated_pipe = hough_circle[0] # 取所有行的第 0 列元素
    center = (calibrated_pipe[0],calibrated_pipe[1])
    radius = calibrated_pipe[2]
    pipe = np.zeros((radius * 2, radius * 2),dtype=np.uint8)
    cv2.circle(pipe, (radius, radius), radius, 255, -1)
    cv2.floodFill(pipe, None, (radius, radius), 255, 0,1)
    cv2.circle(mask, center, radius, (0,255,0), 2)
    cv2.circle(mask, center, 2, (0,0,255), 3)
    # cv2.imshow("pipe", pipe)
    return mask, pipe

'''设置 mindist，确保生成 1 个拟合管道的拟合方式'''
def PipeCircle(frame):
    mindist = max(frame.shape[0], frame.shape[1])
    mask = frame.copy()
    canny = Preprocess(frame)
    circles = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT, edge_config.get("hough_dp"), mindist,
                            param1=edge_config.get("canny_threshold1"),
                            param2=edge_config.get("canny_threshold2"),
                            minRadius=edge_config.get("hough_min_radius"),
                            maxRadius=edge_config.get("hough_max_radius"))
    if circles is None:
        return PipeCenterRestrictor(frame)
    circles = np.uint16(np.around(circles))
    circles = circles.reshape((circles.shape[1], circles.shape[2]))
    center_distance = pixDis(circles[0][0], circles[0][1], frame.shape[1]/2, frame.shape[0]/2)
    if(center_distance > 50): 
        return PipeCenterRestrictor(frame)
    # drawCircle(frame, circles)
    return ReturnedCircle(mask, circles)

'''
限制缺陷中心点的标定方式
'''
def PipeRadiusRestrictor(frame, defect):
    canny = Preprocess(frame)
    defect_center_x = (defect[1] + defect[3]) / 2
    defect_center_y = (defect[2] + defect[4]) / 2
    distance = pixDis(defect_center_x, defect_center_y, frame.shape[1]/2, frame.shape[0]/2)
    mask = frame.copy()
    circles = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT, edge_config.get("hough_dp"), 5,
                            param1=edge_config.get("canny_threshold1"),param2=edge_config.get("canny_threshold2"),
                            minRadius=(int)(distance) - 50, maxRadius=int(distance) + 50)
    if circles is None:
        return PipeCircle(frame)
    circles = np.uint16(np.around(circles))
    # reshape [1, i, 3] to [i, 3]
    circles = circles.reshape((circles.shape[1], circles.shape[2]))
    circles = sorted(circles, key=lambda circle:abs(circle[2] - distance))
    circles = circles[0:len(circles)-1:100]
    # drawCircle(frame, circles)
    return ReturnedCircle(mask, circles)

''' 依据距离图像中心最近的排序方式限制标定管道位置 '''
def PipeCenterRestrictor(frame):
    mask = frame.copy()
    canny = Preprocess(frame)
    circles = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT, edge_config.get("hough_dp"), 5,
                            param1=edge_config.get("canny_threshold1"),
                            param2=edge_config.get("canny_threshold2"),
                            minRadius=edge_config.get("hough_min_radius"),
                            maxRadius=edge_config.get("hough_max_radius"))
    if circles is None:
        return ReturnedCircle(mask, [])
    circles = np.uint16(np.around(circles))
    circles = circles.reshape((circles.shape[1], circles.shape[2]))
    circles = sorted(circles, key=lambda circle:pixDis(circle[0], circle[1], frame.shape[1]/2, frame.shape[0]/2))
    circles = circles[0:len(circles)-1:100]
    # drawCircle(frame, circles)
    '''
    circles[0] is the calibrated base value.
    return value.(i[0], i[1]) is center, and i[2] is radius.
    the following code for draw the calibrated pipe circle in the copy image.
    '''
    return ReturnedCircle(mask, circles)

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
    match = 0
    for file in images:
        print('==>Processing: ', file)
        image = cv2.imread(file)
        defects, features = yolo.detect_image(opencvToPIllow(image))
        result = pil2Opencv(defects)
        detected = []
        if features != None:
            if len(features) == 1:
                feat = features[0]
                mask, pipe = PipeRadiusRestrictor(image, feat)
            else:
                mask, pipe = PipeCircle(image)
            for feat in features:
                dft = Defect(feat[0], image[feat[1]:feat[3], feat[2]:feat[4]])
                detected.append(dft)
                thinker = Thinker(pipe, detected)
                thinker.defect_proportion()
                # savepath = 'datasets/level-sewer10/bx/' + osp.basename(file).split('.')[0] + '-' + str(thinker.level) + '.jpg'
                # print('[SUCCESS][SAVE] ', savepath)
                # save_image(savepath, image)
                if thinker.level == int(osp.basename(file).split("-")[1].split(".")[0]):
                    match += 1
                    # save_image('datasets/01-tmp/success/' + osp.basename(file), image)
                    print('match')
                    break
        else:
            ''' Yolo fail to detect defects '''
            print("[ERROR][YOLO] yolo fail to detect")
            mask, pipe = PipeCircle(image)
            result = image.copy()
            pipeArea = pipe.sum() / 255
            blured = cv2.GaussianBlur(image, (7, 7), 1)
            grayed = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
            imgCanny = cv2.Canny(grayed, edge_config.get("canny_threshold1"), edge_config.get("canny_threshold2"))
            kernel = np.ones((5, 5))
            imgDil = cv2.dilate(imgCanny, kernel, iterations=1) 
            min_area = edge_config.get("min_area")
            contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    cv2.drawContours(result, cnt, -1, (255, 0, 255), 2)
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    # print the quantity of points
                    # print(len(approx))
                    x , y , w, h = cv2.boundingRect(approx)
                    cv2.rectangle(result, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

                    prop = area/pipeArea
                    level = 0
                    cv2.putText(result, "Percent: " + str(prop),(x + w + 20, y + 20),
                                cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2 )
                    if prop > max(dft_rank_tbl['default']['percent']):
                        level = max(dft_rank_tbl['default']['level'])
                        cv2.putText(result, "Level: " + str(level),(x + w + 20, y + 45),
                                cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2 )
                        break
                    for i, percent in enumerate(dft_rank_tbl['default']['percent']):
                        if (prop < percent):
                            level = i
                            cv2.putText(result, "Level: " + str(level),(x + w + 20, y + 45),
                                    cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2 )
                            break
                    # savepath = 'datasets/level-sewer10/bx/' + osp.basename(file).split('.')[0] + '-' + str(level) + '.jpg'
                    # print('[SUCCESS][SAVE] ', savepath)
                    # save_image(savepath, image)
                    if thinker.level == int(osp.basename(file).split("-")[1].split(".")[0]):
                        match += 1
                        # save_image('datasets/01-tmp/success/' + osp.basename(file), image)
                        print('match')
                        break
        
        '''display results'''
        # imgStack = stackImages(0.3, ([mask, result]))
        # cv2.imshow(window_caption, imgStack)
        # cv2.waitKey()
    print("准确率: ", match / len(images))
    cv2.destroyAllWindows()
    return match, match / len(images)

def ShowDatasetsPipe(path):
    print("Ready to Loading datasets: ",osp.abspath(path))
    images = set()
    categories = os.listdir(path);
    if '.DS_Store' in categories:
        categories.remove('.DS_Store')
        '''for err-sample dir'''
        # categories.remove('README.md')
    for category in categories:
        images = set()
        c_path = osp.join(path, category)
        files = os.listdir(c_path)
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        for file in files:
            filename = osp.join(c_path, file)
            images.add(filename)
        print("========> Imagesets initialized successfully for: ", category)
        match, acc = ShowImages(images)
        print("========> Accuracy for category {:s}: {:3F}, match: {:d}, total: {:d} <========".format(category, acc, match, len(images)))

ShowDatasetsPipe(conf.get("datasets"))
