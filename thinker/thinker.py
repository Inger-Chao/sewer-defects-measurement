import cv2
import math
import numpy as np
from config import edge_config

class Defect:
    def __init__(self, name, image) -> None:
        self.name = name
        self.image = image
    
    def get_rank(self):
        res = 0
        return 0

class Thinker:
    def __init__(self, pipe, defects) -> None:
        self.pipe = pipe
        '''
        defects: map, [class, detected_area]
        '''
        self.defects = defects

    def defect_area(self, image):
        blured = cv2.GaussianBlur(image, (7, 7), 1)
        grayed = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(grayed, edge_config.get("canny_threshold1"), edge_config.get("canny_threshold2"))
        kernel = np.ones((5, 5))
        dil = cv2.dilate(canny, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > edge_config.get("min_area"):
                cv2.drawContours(image, cnt, -1, (255, 0, 255), 7)
                return area
    
    def defect_proportion(self):
        pipe_base = self.pipe.sum() / 255
        for feature in self.defects:
            name = feature.name
            area = self.defect_area(feature.image)
            print('Detected class: ', name, ' Defect Rank: ', area/pipe_base)
