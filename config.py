# -*- coding: utf-8 -*-
# @Time    : 2020-11-13 19:14
# @Author  : Inger

conf = {
    "path": "datasets/json",
    "datasets": "datasets/sewer10/zhaw"
}

yolo_config = {
    "model_path"        : 'dl_model/Epoch300-Total_Loss3.0036-Val_Loss7.3584.pth',
    "anchors_path"      : 'dl_model/yolo_anchors.txt',
    "classes_path"      : 'dl_model/DEFECTS.txt',
    "model_image_size"  : (416, 416, 3),
    "confidence"        : 0.5,
    "iou"               : 0.3,
    # GPU: True, CPU: False
    "cuda"              : False,
    # resize image
    "letterbox_image"   : False,
}

edge_config = {
    # canny threshold param
    "canny_threshold1": 142,
    "canny_threshold2": 146,
    "coefficient": 4,
    "min_area": 9220,
    "hough_dp": 2,
    "hough_min_dist": 480,
    "hough_min_radius": 64,
    "hough_max_radius": 640
}