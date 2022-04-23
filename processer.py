# -*- coding: utf-8 -*-
# @Time    : 2020-11-13 19:09
# @Author  : Inger

import imghdr
import os
from pickle import FALSE
from pydoc import pipepager
import cv2
from cv2 import blur
import numpy as np
from data_loader import DataLoader, load_songbai_data
from pipe_calibration import PipeCenterRestrictor, PipeCircle
from utils.cv_util import stackImages
from utils.utils import opencvToPIllow, pil2Opencv
from detector.defects_detector import YOLO
from nets.hed import HedLayer
from config import edge_config, dft_rank_tbl

def empty(a):
    pass

canny_threshold1 = edge_config.get("canny_threshold1")
canny_threshold2 = edge_config.get("canny_threshold2")
coefficient = edge_config.get("coefficient")
min_area = edge_config.get("min_area")

yolo = YOLO()

str_parameters="methodology"
cv2.namedWindow("Parameters")
cv2.namedWindow(str_parameters)
cv2.resizeWindow("Parameters",640,240)
# cv2.namedWindow("APC")
# cv2.resizeWindow("APC", 640, 240)
cv2.createTrackbar("erode", str_parameters, 3, 6, empty)
cv2.createTrackbar("dialte", str_parameters, 5, 6, empty)
cv2.createTrackbar("erode_times", str_parameters, 1, 5, empty)
cv2.createTrackbar("dialte_times", str_parameters, 2, 5, empty)
cv2.createTrackbar("Threshold1","Parameters", canny_threshold1,255, empty)
cv2.createTrackbar("Threshold2","Parameters", canny_threshold2,255, empty)
cv2.createTrackbar("Coefficient", "Parameters", coefficient, 20, empty)
cv2.createTrackbar("Area","Parameters", min_area,30000, empty)


def defectsLevel(defectArea, pipeArea):
    defects_ratio = defectArea / pipeArea
    if(defects_ratio < 0.1):
        return "Slight"
    elif(defects_ratio <= 0.5):
        return "Middle"
    else:
        return "Serious"

def getContours(img, w, pipe,imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # pipeCnt, _ = cv2.findContours(pipe, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # pipeArea = (w/10.0) * cv2.contourArea(pipeCnt[0])
    pipeArea = pipe.sum()/128
    min_area = cv2.getTrackbarPos("Area","Parameters")
    total_level = 0
    level = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 4)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print the quantity of points
            # print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

            prop = area/pipeArea
            if prop > max(dft_rank_tbl['default']['percent']):
                level = max(dft_rank_tbl['default']['level'])
                total_level = max(level, total_level)
                # cv2.putText(imgContour, "Level: " + str(level),(x + 15, y + 20),
                        # cv2.FONT_HERSHEY_COMPLEX, .7,(0, 0, 255), 2 )
                break
            for i, percent in enumerate(dft_rank_tbl['default']['percent']):
                if (prop < percent):
                    level = i
                    total_level = max(level, total_level)
                    # cv2.putText(imgContour, "Level: " + str(level),(x + 15, y+20),
                    #         cv2.FONT_HERSHEY_COMPLEX, .7,(0, 0, 255), 2 )
                    prop = round(prop, 3)
                    # cv2.putText(imgContour, "Prop: " + str(prop),(x + 15, y+45),
                            # cv2.FONT_HERSHEY_COMPLEX, .7,(0, 0, 255), 2 )
                    print("[INFO][PROP] =  " + str(prop) + "  [LEVEL] = " + str(level))
                    break
    return total_level

hed = HedLayer()

def main(yolo_flag=True):
    imgs, pipes = DataLoader()
    # imgs = load_songbai_data()
    for id, file in enumerate(imgs):
        print("[INFO][PROCESSING]=====>", file)
        pipe = pipes.__getitem__(id)
        # level = os.path.basename(file).split('-')[1].split('.')[0]
        img = cv2.imread(file)
        # mask, pipe = PipeCircle(pipe)
        blured = cv2.GaussianBlur(img, (7, 7), 1)
        grayed = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
        imgYolo, defects_feature = yolo.detect_image(opencvToPIllow(img))
        imgYolo = pil2Opencv(imgYolo)
        if yolo_flag is False:
            defects_feature=None
        while True:
            imgCopy = img.copy()
            canny_threshold1 = cv2.getTrackbarPos("Threshold1","Parameters")
            canny_threshold2 = cv2.getTrackbarPos("Threshold2","Parameters")

            coefficient = cv2.getTrackbarPos("Coefficient", "Parameters")
            if defects_feature != None :
                count = 1
                for defect in defects_feature:
                    imgHED = hed.forward(blured)
                    imgCanny = cv2.Canny(blured, canny_threshold1, canny_threshold2)
                    defect_area = img[defect[1]:defect[3], defect[2]:defect[4]]
                    defect_copy = defect_area.copy()
                    defect_gray = grayed[defect[1]:defect[3], defect[2]:defect[4]]
                    # defect_dil = imgDil[defect[1]:defect[3], defect[2]:defect[4]]
                    _, defect_dil = cv2.threshold(src=defect_gray,thresh=64,maxval=255, type=cv2.THRESH_OTSU)
                    cv2.imshow("defect area" + str(count), defect_dil)
                    cmp_level = getContours(defect_dil, coefficient, pipe, defect_copy)
                    imgCopy[defect[1]:defect[3], defect[2]:defect[4]] = defect_copy
                    count += 1
                    imgStack = stackImages(0.4, ([imgYolo, imgCopy]))
            else:
                imgHED = hed.forward(blured)
                _,otsu_hed = cv2.threshold(src=imgHED, thresh=64, maxval=255, type=cv2.THRESH_OTSU)
                # cv2.imshow("otsu hed", otsu_hed)
                erode_kernel = cv2.getTrackbarPos("erode", str_parameters)
                dialte_kernel = cv2.getTrackbarPos("dialte", str_parameters)
                erode_times = cv2.getTrackbarPos("erode_times", str_parameters)
                dialte_times = cv2.getTrackbarPos("dialte_times", str_parameters)
                imgCanny = cv2.Canny(imgHED, canny_threshold1, canny_threshold2)
                imgErode1 = cv2.erode(imgCanny, kernel=np.ones((2,2)), iterations=1)
                imgDil = cv2.dilate(imgErode1, np.ones((dialte_kernel, dialte_kernel)), iterations=dialte_times)
                imgErode = cv2.erode(imgDil, kernel=np.ones((erode_kernel,erode_kernel)), iterations=erode_times)
                cmp_level = getContours(otsu_hed, coefficient, pipe, imgCopy)
                imgStack = stackImages(0.4, ([imgYolo, imgHED],
                                            [imgCanny, imgErode1],
                                            [imgDil, imgErode],
                                            [pipe, imgCopy]))
            cv2.imshow("Parameters", imgStack)
            if cv2.waitKey() & 0xFF == ord("q"):
                break
            if cv2.waitKey() & 0xFF == ord("w"):
                with open("result_log/1104-songbai-parameters.txt", 'a') as log_file:
                    min_area = cv2.getTrackbarPos("Area","Parameters")
                    txt = '{%s}\t{%d}\t{%d}\t{%d}\n' % (file, canny_threshold1, canny_threshold2, min_area)
                    log_file.write(txt)
                break
            # recomputing
            if cv2.waitKey() & 0xFF == ord("r"):
                continue
    cv2.destroyAllWindows()

main(yolo_flag=False)