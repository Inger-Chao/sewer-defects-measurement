import cv2
from cv2 import circle
from matplotlib.pyplot import draw
import numpy as np

from pipe_calibration import Preprocess, ShowImages, drawCircle, PipeCenterRestrictor
from config import edge_config
from utils.cv_util import pixDis

def reshapeCircles(circles):
    circles = np.uint16(np.around(circles))
    circles = circles.reshape((circles.shape[1], circles.shape[2]))
    circles = circles[0:len(circles)-1:100]
    return circles
'''大论文里面画图的'''
def HoughSamples(frame):
    mindist = max(frame.shape[0], frame.shape[1])
    mask = frame.copy()
    canny = Preprocess(frame)
    # circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 20, param1=edge_config.get("canny_threshold1"), 
    # param2=edge_config.get("acc_threshold"), minRadius=edge_config.get("hough_min_radius"),
    #                         maxRadius=edge_config.get("hough_max_radius"))
    # drawCircle(frame, reshapeCircles(circles))
    # cv2.waitKey(0)
    # circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 2, 20, param1=edge_config.get("canny_threshold1"), 
    # param2=edge_config.get("acc_threshold"), minRadius=edge_config.get("hough_min_radius"),
    #                         maxRadius=edge_config.get("hough_max_radius"))
    # drawCircle(frame, reshapeCircles(circles))
    # cv2.waitKey(0)
    # circles = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT, edge_config.get("hough_dp"), mindist,
    #                         param1=edge_config.get("canny_threshold1"),
    #                         param2=edge_config.get("acc_threshold"),
    #                         minRadius=edge_config.get("hough_min_radius"),
    #                         maxRadius=edge_config.get("hough_max_radius"))
    # circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 20, param1=edge_config.get("canny_threshold1"), 
    # param2=edge_config.get("acc_threshold"), minRadius=mindist,maxRadius=100)
    # if circles is None:
    #     return PipeCenterRestrictor(frame)
    # circles = reshapeCircles(circles)
    # center_distance = pixDis(circles[0][0], circles[0][1], frame.shape[1]/2, frame.shape[0]/2)
    # if(center_distance > 50): 
    #     return PipeCenterRestrictor(frame)
    # cv2.waitKey(0)

# 障碍物缺陷
# img = cv2.imread("/Users/inger/underground/pic/latex/apc/orginal.png")
# 无缺陷管道样本
# img = cv2.imread("/Users/inger/underground/资料/Image2/000009.jpg")
# HoughSamples(img)
# ShowImages(["/Users/inger/underground/pic/latex/apc/orginal.png"])

import sys
import cv2
import math
import numpy
from scipy.ndimage import label
pi_4 = 4*math.pi

def nothing_asCallback(x):
    pass

# dp参数可视化
def GUI_openCV_circles():
    # --------------------------------------------------------------------------------GUI-<image>
    frame = cv2.imread(  "/Users/inger/underground/资料/Image2/000009.jpg" )
    demo  = frame[:800,:800,:]
    # --------------------------------------------------------------------------------GUI-<window>-s
    cv2.namedWindow( "DEMO.IN" )
    cv2.namedWindow( "DEMO.Canny")
    cv2.namedWindow( "DEMO.Canny.Circles")
    # --------------------------------------------------------------------------------GUI-<state>-initial-value(s)
    aKeyPRESSED                                     = None              # .init

    aCanny_LoTreshold                               = 127
    aCanny_LoTreshold_PREVIOUS                      =  -1
    aCanny_HiTreshold                               = 255
    aCanny_HiTreshold_PREVIOUS                      =  -1

    aHough_dp                                       =   1
    aHough_dp_PREVIOUS                              =  -1
    aHough_minDistance                              =  10
    aHough_minDistance_PREVIOUS                     =  -1
    aHough_param1_aCannyHiTreshold                  = 255
    aHough_param1_aCannyHiTreshold_PREVIOUS         =  -1
    aHough_param2_aCentreDetectTreshold             =  20
    aHough_param2_aCentreDetectTreshold_PREVIOUS    =  -1
    aHough_minRadius                                =  10
    aHough_minRadius_PREVIOUS                       =  -1
    aHough_maxRadius                                =  30
    aHough_maxRadius_PREVIOUS                       =  -1
    # --------------------------------------------------------------------------------GUI-<ACTOR>-s
    cv2.createTrackbar( "Lo_Treshold",          "DEMO.Canny",          aCanny_LoTreshold,                      255, nothing_asCallback )
    cv2.createTrackbar( "Hi_Treshold",          "DEMO.Canny",          aCanny_HiTreshold,                      255, nothing_asCallback )

    cv2.createTrackbar( "dp",                   "DEMO.Canny.Circles",  aHough_dp,                              255, nothing_asCallback )
    cv2.createTrackbar( "minDistance",          "DEMO.Canny.Circles",  aHough_minDistance,                     255, nothing_asCallback )
    cv2.createTrackbar( "param1_HiTreshold",    "DEMO.Canny.Circles",  aHough_param1_aCannyHiTreshold,         255, nothing_asCallback )
    cv2.createTrackbar( "param2_CentreDetect",  "DEMO.Canny.Circles",  aHough_param2_aCentreDetectTreshold,    255, nothing_asCallback )
    cv2.createTrackbar( "minRadius",            "DEMO.Canny.Circles",  aHough_minRadius,                       255, nothing_asCallback )
    cv2.createTrackbar( "maxRadius",            "DEMO.Canny.Circles",  aHough_maxRadius,                       255, nothing_asCallback )

    cv2.imshow( "DEMO.IN",          demo )                              # static ...
    # --------------------------------------------------------------------------------GUI-mainloop()
    print(" --------------------------------------------------------------------------- press [ESC] to exit ")
    while( True ):
        # --------------------------------------------------------------------------------GUI-[ESCAPE]?
        if aKeyPRESSED == 27:
            break
        # --------------------------------------------------------------------------------<vars>-DETECT-delta(s)
        aCanny_LoTreshold = cv2.getTrackbarPos( "Lo_Treshold", "DEMO.Canny" )
        aCanny_HiTreshold = cv2.getTrackbarPos( "Hi_Treshold", "DEMO.Canny" )

        if (    aCanny_LoTreshold      != aCanny_LoTreshold_PREVIOUS
            or  aCanny_HiTreshold      != aCanny_HiTreshold_PREVIOUS
            ):
            # --------------------------= FLAG
            aCannyRefreshFLAG           = True
            # --------------------------= RE-SYNC
            aCanny_LoTreshold_PREVIOUS  = aCanny_LoTreshold
            aCanny_HiTreshold_PREVIOUS  = aCanny_HiTreshold
        else:
            # --------------------------= Un-FLAG
            aCannyRefreshFLAG           = False

        aHough_dp                           = cv2.getTrackbarPos( "dp",                 "DEMO.Canny.Circles" )
        aHough_minDistance                  = cv2.getTrackbarPos( "minDistance",        "DEMO.Canny.Circles" )
        aHough_param1_aCannyHiTreshold      = cv2.getTrackbarPos( "param1_HiTreshold",  "DEMO.Canny.Circles" )
        aHough_param2_aCentreDetectTreshold = cv2.getTrackbarPos( "param2_CentreDetect","DEMO.Canny.Circles" )
        aHough_minRadius                    = cv2.getTrackbarPos( "minRadius",          "DEMO.Canny.Circles" )
        aHough_maxRadius                    = cv2.getTrackbarPos( "maxRadius",          "DEMO.Canny.Circles" )

        if (    aHough_dp                            != aHough_dp_PREVIOUS
            or  aHough_minDistance                   != aHough_minDistance_PREVIOUS
            or  aHough_param1_aCannyHiTreshold       != aHough_param1_aCannyHiTreshold_PREVIOUS
            or  aHough_param2_aCentreDetectTreshold  != aHough_param2_aCentreDetectTreshold_PREVIOUS    
            or  aHough_minRadius                     != aHough_minRadius_PREVIOUS
            or  aHough_maxRadius                     != aHough_maxRadius_PREVIOUS
            ):
            # --------------------------= FLAG
            aHoughRefreshFLAG           = True                  
            # ----------------------------------------------= RE-SYNC
            aHough_dp_PREVIOUS                              =  aHough_dp                          
            aHough_minDistance_PREVIOUS                     =  aHough_minDistance                 
            aHough_param1_aCannyHiTreshold_PREVIOUS         =  aHough_param1_aCannyHiTreshold     
            aHough_param2_aCentreDetectTreshold_PREVIOUS    =  aHough_param2_aCentreDetectTreshold
            aHough_minRadius_PREVIOUS                       =  aHough_minRadius                   
            aHough_maxRadius_PREVIOUS                       =  aHough_maxRadius                   
        else:
            # --------------------------= Un-FLAG
            aHoughRefreshFLAG           = False
        # --------------------------------------------------------------------------------REFRESH-process-pipe-line ( with recent <state> <vars> )
        if ( aCannyRefreshFLAG ):

            edges   = cv2.Canny(        demo,   aCanny_LoTreshold,
                                                aCanny_HiTreshold
                                        )
            # --------------------------------------------------------------------------------GUI-SHOW-Canny()-<edges>-onRefreshFLAG
            cv2.imshow( "DEMO.Canny",   edges )
            pass

        if ( aCannyRefreshFLAG or aHoughRefreshFLAG ):

            circles = cv2.HoughCircles( edges,  cv2.HOUGH_GRADIENT,
                                                aHough_dp,
                                                aHough_minDistance,
                                                param1      = aHough_param1_aCannyHiTreshold,
                                                param2      = aHough_param2_aCentreDetectTreshold,
                                                minRadius   = aHough_minRadius,
                                                maxRadius   = aHough_maxRadius
                                        )
            # --------------------------------------------------------------------------------GUI-SHOW-HoughCircles()-<edges>-onRefreshFLAG
            demoWithCircles = cv2.cvtColor( demo,            cv2.COLOR_BGR2RGB )                          # .re-init <<< src
            demoWithCircles = cv2.cvtColor( demoWithCircles, cv2.COLOR_RGB2BGR )

            for aCircle in circles[0]:
                cv2.circle( demoWithCircles,    ( int( aCircle[0] ), int( aCircle[1] ) ),
                                                aCircle[2],
                                                (0,255,0),
                                                1
                            )
                pass
            pass
            cv2.imshow( "DEMO.Canny.Circles", demoWithCircles )
        pass        
        # --------------------------------------------------------------------------------<vars>-UPDATE-<state>
        # ref. above in .onRefreshFLAG RE-SYNC sections
        # --------------------------------------------------------------------------------GUI-INPUT ? [ESCAPE]
        aKeyPRESSED = cv2.waitKey(1) & 0xFF
    pass
    # --------------------------------------------------------------------------------GUI-<window>-s / DESTROY
    cv2.destroyWindow( "DEMO.IN" )
    cv2.destroyWindow( "DEMO.Canny" )
    cv2.destroyWindow( "DEMO.Canny.Circles" )
    # --------------------------------------------------------------------------------GUI-<window>-s
    pass

def main():
    GUI_openCV_circles()
    return 0

if __name__ == '__main__':
    main()