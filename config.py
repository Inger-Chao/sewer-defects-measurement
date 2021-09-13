# -*- coding: utf-8 -*-
# @Time    : 2020-11-13 19:14
# @Author  : Inger

conf = {
    "path": "datasets/json",
    "classifier": "cascade",
    "coefficient": 1.2,
    "minArea": 40000
}

yolo_config = {
    "model_path"        : 'dl_model/Epoch300-Total_Loss3.0036-Val_Loss7.3584.pth',
    "anchors_path"      : 'dl_model/yolo_anchors.txt',
    "classes_path"      : 'dl_model/DEFECTS.txt',
    "model_image_size"  : (416, 416, 3),
    "confidence"        : 0.8,
    "iou"               : 0.3,
    # GPU: True, CPU: False
    "cuda"              : False,
    # resize image
    "letterbox_image"   : False,
}