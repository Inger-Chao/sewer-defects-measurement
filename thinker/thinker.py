import cv2
import math
import numpy as np
from numpy.core.defchararray import index
from nets.hed import HedLayer
from utils.cv_util import empty, stackImages
from config import edge_config, dft_rank_tbl

class Defect:
    def __init__(self, name, image) -> None:
        self.name = name
        self.image = image
    
    def get_rank(self):
        self.rank = dft_rank_tbl[self.name]
        res = 0
        return res

class Thinker:
    def __init__(self, pipe, defects) -> None:
        self.pipe = pipe
        '''
        defects: map: [class, detected_area]
        '''
        self.defects = defects

    def defect_area(self, image):
        mask = image.copy()
        blured = cv2.GaussianBlur(mask, (7, 7), 1)
        grayed = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
        # kernel = np.ones((1, 1))
        # dil = cv2.dilate(canny, kernel, iterations=1)
        _,thresh1 = cv2.threshold(src=grayed, thresh=edge_config['bin_thresh'],maxval=255, type=cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        maxarea = 0
        sumaraea = 0
        try:
            maxarea = max(cv2.contourArea(cnt) for cnt in contours)
            sumaraea = sum(cv2.contourArea(cnt) for cnt in contours)
        except:
            print('find contour area error')
        # for cnt in contours:
        #     if cv2.contourArea(cnt) > edge_config['defect_min_area']:
        #         cv2.drawContours(mask, cnt, -1, (255, 0, 255), 1)
        # stack_image = stackImages(0.5, [[mask, thresh1]])
        # cv2.imshow("defect feature", stack_image)
        return maxarea, sumaraea
    
    def defect_proportion(self):
        pipe_base = self.pipe.sum() / 255
        self.level = 0
        for feature in self.defects:
            name = feature.name.lower()
            area1, area2 = self.defect_area(feature.image)
            prop1 = np.float(area1 / pipe_base)
            prop2 = np.float(area2 / pipe_base)
            try:            
                for i, percent in enumerate(dft_rank_tbl[name]['percent']):
                    if (prop2 < percent):
                        self.level = i
                        # print('Detected class: ', name, ' Defect prop2: ', prop2, ' level: ', self.level)
                        break
                    if (prop1 < percent):
                        self.level = i
                        # print('Detected class: ', name, ' Defect prop1: ', prop1, ' level: ', self.level)
                        continue
                if prop1 > max(dft_rank_tbl[name]['percent']):
                    self.level = max(dft_rank_tbl[name]['level'])
            except:
                print('[ERROR][YOLO] Unsupported defect class: ', name)

class EntireProcesser:
    def __init__(self, pipe, image) -> None:
        self.pipe = pipe
        self.image = image
    
    def entire_image(self):
        pipe_base = self.pipe.sum() / 255
        self.level = 0
        HED_layer = HedLayer()
        result = self.image.copy()
        blured = cv2.GaussianBlur(self.image, (7, 7), 1)
        # grayed = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
        hed = HED_layer.forward(blured)
        # cv2.imshow("hed", hed)
        imgCanny = cv2.Canny(hed, edge_config.get("canny_threshold1"), edge_config.get("canny_threshold2"))
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

                prop = area/pipe_base
                cv2.putText(result, "Percent: " + str(prop),(x + w + 20, y + 20),
                            cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2 )
                if prop > max(dft_rank_tbl['default']['percent']):
                    self.level = max(dft_rank_tbl['default']['level'])
                    cv2.putText(result, "Level: " + str(self.level),(x + w + 20, y + 45),
                            cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2 )
                    break
                for i, percent in enumerate(dft_rank_tbl['default']['percent']):
                    if (prop < percent):
                        self.level = i
                        cv2.putText(result, "Level: " + str(self.level),(x + w + 20, y + 45),
                                cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2 )
                        break
        return result

    
    def process_level(self, true_level=0):
        pipe_base = self.pipe.sum() / 255
        self.level = 0
        result = self.image.copy()
        blured = cv2.GaussianBlur(self.image, (7, 7), 1)
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

                prop = area/pipe_base
                cv2.putText(result, "Percent: " + str(prop),(x + w + 20, y + 20),
                            cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2 )
                if prop > max(dft_rank_tbl['default']['percent']):
                    self.level = max(dft_rank_tbl['default']['level'])
                    if self.level == true_level:
                        return
                    cv2.putText(result, "Level: " + str(self.level),(x + w + 20, y + 45),
                            cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2 )
                    break
                for i, percent in enumerate(dft_rank_tbl['default']['percent']):
                    if (prop < percent):
                        self.level = i
                        if self.level == true_level:
                            return
                        cv2.putText(result, "Level: " + str(self.level),(x + w + 20, y + 45),
                                cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2 )
                        break
    