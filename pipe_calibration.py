# -*- coding: utf-8 -*-
# @Time    : 2021-09-06 14:49
# @Author  : Inger
from typing import Set
import cv2
import numpy as np
import os
import os.path as osp
import random
from data_loader import videos_path
from nets.hed import HedLayer
from utils.cv_util import save_image, stackImages, pixDis
from utils.utils import opencvToPIllow, pil2Opencv, isImageFile, removeDStore
from detector.defects_detector import YOLO
from config import edge_config, conf, dft_rank_tbl
from thinker.thinker import Defect, EntireProcesser, Thinker


font = cv2.FONT_HERSHEY_COMPLEX
yolo = YOLO()

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
        center = ((int)(mask.shape[1] * 0.5), int(mask.shape[0] * 0.5))
        radius = min((int)(mask.shape[0] * 0.3), (int)(mask.shape[1] * 0.3))
        pipe = np.zeros((radius * 2, radius * 2),dtype=np.uint8)
        cv2.circle(pipe, (radius, radius), radius, 255, -1)
        cv2.circle(mask, center, radius, (0,255,0), 2)
        cv2.circle(mask, center, 2, (0,0,255), 3)
        return mask, pipe
    calibrated_pipe = hough_circle[0] # 取所有行的第 1 个元素为最优解
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
                            param2=edge_config.get("acc_threshold"),
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
                            param1=edge_config.get("canny_threshold1"),param2=edge_config.get("acc_threshold"),
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
    circles = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT, edge_config.get("hough_dp"), 
                            (int)(frame.shape[0]/random.uniform(5, 20)),
                            param1=edge_config.get("canny_threshold1"),
                            param2=edge_config.get("acc_threshold"),
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
    print("[INFO][PROCESSING]==>", video_path)
    cap = cv2.VideoCapture(video_path)
    
    #读取视频的fps,  大小
    fps=cap.get(cv2.CAP_PROP_FPS)
    size=(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("fps: {}\nsize: {}".format(fps,size))
    
    #读取视频时长（帧总数）
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] {} total frames in video".format(total))
    while True:
        ret, frame = cap.read()
        detected = []
        try:
            frame = frame[100:frame.shape[0]-50, 50:frame.shape[1]]
        except:
            break
        mask, pipe = PipeCircle(frame)
        # result, features = yolo.detect_image(opencvToPIllow(frame))
        # result = pil2Opencv(result)
        # if features is None:
        thinker = EntireProcesser(pipe, frame)
        result = thinker.entire_image()
        # else:
        #     for feat in features:
        #         dft = Defect(feat[0], frame[feat[1]:feat[3], feat[2]:feat[4]])
        #         detected.append(dft)
        #         thinker = Thinker(pipe, detected)
        #         thinker.defect_proportion()
        #     result = frame

        imgStack = stackImages(0.5, ([mask, result]))
        cv2.imshow(window_caption, imgStack)

        #--------键盘控制视频---------------
        #读取键盘值
        key = cv2.waitKey(1) & 0xff
        #设置空格按下时暂停
        if key == ord(" "):
            cv2.waitKey(0)
        #设置Q按下时退出
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def ShowImages(images):
    match = 0
    for file in images:
        # print('==>Processing: ', file)
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
                    # print('match')
                    break
        else:
            ''' Yolo fail to detect defects '''
            print("[ERROR][YOLO] yolo fail to detect")
            mask, pipe = PipeCircle(image)
            true_level = int(osp.basename(file).split("-")[1].split(".")[0])
            thinker = EntireProcesser(pipe, image)
            thinker.process_level(true_level)
            result = thinker.entire_image()
            # savepath = 'datasets/level-sewer10/bx/' + osp.basename(file).split('.')[0] + '-' + str(level) + '.jpg'
            # print('[SUCCESS][SAVE] ', savepath)
            # save_image(savepath, image)
            if thinker.level == true_level:
                match += 1
                # save_image('datasets/01-tmp/success/' + osp.basename(file), image)
                # print('match')
                # break

        '''display results'''
        imgStack = stackImages(0.3, ([mask, result]))
        cv2.imshow(window_caption, imgStack)
        cv2.waitKey()
    cv2.destroyAllWindows()
    if (len(images)==0):
        return match, 0
    return match, round(match / len(images), 3)

def ShowDatasets(path):
    print("Ready to Loading datasets: ",osp.abspath(path))
    res_tables = np.zeros((5, 15))
    acc_tables = np.zeros((5, 15))
    categories = os.listdir(path);
    removeDStore(categories)
    for type in categories:
        c_type_dir = osp.join(path, type)
        c_level_dirs = os.listdir(c_type_dir)
        removeDStore(c_level_dirs)
        for level in c_level_dirs:
            images = list()
            files_path = osp.join(c_type_dir, level)
            filenames = os.listdir(files_path)
            removeDStore(filenames)
            for image in filenames:
                images.append(osp.join(files_path, image))
            # print("========> Imagesets initialized successfully for: ", type)
            match, acc = ShowImages(images)
            res_tables[int(level)][dft_rank_tbl[type]['id']] = match
            acc_tables[int(level)][dft_rank_tbl[type]['id']] = acc
            print("========> Accuracy for category {:s}: {:3F} with level {:s}, match: {:d}, total: {:d} <========".format(type, acc, level ,match, len(images)))
    return res_tables, acc_tables

def getAllLevelAP(path):
    print("Ready to Loading datasets: ",osp.abspath(path))
    res_tables = np.zeros((5, 15))
    acc_tables = np.zeros((5, 15))
    categories = os.listdir(path);
    removeDStore(categories)
    for type in categories:
        c_type_dir = osp.join(path, type)
        c_level_dirs = os.listdir(c_type_dir)
        removeDStore(c_level_dirs)
        images = list()
        for level in c_level_dirs:
            files_path = osp.join(c_type_dir, level)
            filenames = os.listdir(files_path)
            removeDStore(filenames)
            for image in filenames:
                images.append(osp.join(files_path, image))
            # print("========> Imagesets initialized successfully for: ", type)
        match, acc = ShowImages(images)
        res_tables[int(level)][dft_rank_tbl[type]['id']] = match
        acc_tables[int(level)][dft_rank_tbl[type]['id']] = acc
        print("========> Accuracy for category {:s}: {:3F} with all level, match: {:d}, total: {:d} <========".format(type, acc,match, len(images)))
    return res_tables, acc_tables