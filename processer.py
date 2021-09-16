# -*- coding: utf-8 -*-
# @Time    : 2020-11-13 19:09
# @Author  : Inger

import cv2
import numpy as np
from data_loader import DataLoader
from utils.cv_util import stackImages
from utils.utils import opencvToPIllow, pil2Opencv
from detector.defects_detector import YOLO
from config import edge_config

def empty(a):
    pass

canny_threshold1 = edge_config.get("canny_threshold1")
canny_threshold2 = edge_config.get("canny_threshold2")
coefficient = edge_config.get("coefficient")
min_area = edge_config.get("min_area")

yolo = YOLO()

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
# cv2.namedWindow("APC")
# cv2.resizeWindow("APC", 640, 240)
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
    pipeCnt, _ = cv2.findContours(pipe, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    pipeArea = (w/10.0) * cv2.contourArea(pipeCnt[0])
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print the quantity of points
            # print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

            # cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
            #             (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, "Level: " + str(area/pipeArea),(x + w + 20, y + 20),
                        cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2 )
            print("Level: " + str(area/pipeArea))


def main():
    imgs, pipes = DataLoader()
    for id, img in enumerate(imgs):
        pipe = pipes.__getitem__(id)
        blured = cv2.GaussianBlur(img, (7, 7), 1)
        grayed = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
        grayedPipe = cv2.cvtColor(pipe, cv2.COLOR_BGR2GRAY)
        imgYolo, defects_feature = yolo.detect_image(opencvToPIllow(img))
        imgYolo = pil2Opencv(imgYolo)
        imgCopy = img.copy()
        while True:
            imgCanny = cv2.Canny(grayed, canny_threshold1, canny_threshold2)
            kernel = np.ones((5, 5))
            img_not = cv2.bitwise_not(imgCanny)

            coefficient = cv2.getTrackbarPos("Coefficient", "Parameters")
            if defects_feature != None :
                for defect in defects_feature:
                    defect_area = img[defect[1]:defect[3], defect[2]:defect[4]]
                    defect_copy = defect_area.copy()
                    cv2.imshow("defect area", defect_area)
                    defect_canny = cv2.Canny(defect_area, canny_threshold1, canny_threshold2)
                    
                    defect_dil = cv2.dilate(defect_canny, kernel, iterations=1)
                    getContours(defect_dil, coefficient, grayedPipe, defect_copy)
                    imgCopy[defect[1]:defect[3], defect[2]:defect[4]] = defect_copy
            else:
                imgDil = cv2.dilate(imgCanny, kernel, iterations=1) 
                getContours(imgDil, coefficient, grayedPipe, imgCopy)
            imgStack = stackImages(0.4, ([img, imgCanny],
                                            [img_not, imgCopy]))
            cv2.imshow("Parameters", imgStack)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

main()