# -*- coding: utf-8 -*-
# @Time    : 2020-11-13 19:14
# @Author  : Inger

conf = {
    "path": "datasets/json",
    "datasets": "datasets/success"
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
    "min_area": 8000,
    "hough_dp": 2,
    "hough_min_dist": 480,
    "hough_min_radius": 32,
    "hough_max_radius": 640,
    "defect_min_area": 255,
    "bin_thresh": 155
}

dft_rank_tbl = {
    'default': {
        'id': 1,
        'level': [0, 1, 2, 3, 4],
        'percent': [0, 0.2, 0.4, 0.6, 1],
        'rank': [0, 0.1, 2, 5, 10]
    },
    'chj': {
        'id': 1,
        'level': [0, 1, 2, 3, 4],
        'percent': [0, 0.2, 0.4, 0.6, 1],
        'rank': [0, 0.1, 2, 5, 10]
    },
    'bx': {
        'id': 2,
        'level': [0, 1, 2, 3, 4],
        'percent': [0, 0.05, 0.15, 0.25, 1],
        'rank': [0, 1, 2, 5, 10]
    },
    'zhaw': {
        'id': 3,
        'level': [0, 1, 2, 3, 4],
        'percent': [0, 0.15, 0.25, 0.5, 1],
        'rank': [0, 0.1, 2, 5, 10]
    },
    'qf': {
        'id': 5,
        'level': [0, 1, 2, 3, 4],
        'percent': [0, 0.2, 0.35, 0.5, 1],
        'rank': [0, 0.5, 2, 5, 10]
    },
    'jg': {
        'id': 7,
        'level': [0, 1, 2, 3, 4],
        'percent': [0, 0.15, 0.25, 0.5, 0.8],
        'rank': [0, 0.5, 2, 5, 10]
    },
    'shg': {
        'id': 8,
        'level': [0, 1, 2, 3, 4],
        'percent': [0, 0.15, 0.25, 0.5, 1],
        'rank': [0, 0.5, 2, 5, 10]
    },
    'pl': {
        'id': 9,
        'level': [0, 1, 2, 3, 4],
        'percent': [0, 0.1, 0.25, 0.6, 1],
        'rank': [0, 0.5, 2, 5, 10]
    },
    'zhgaj': {
        'id': 11,
        'level': [0, 1, 2, 3],
        'percent': [0, 0.1, 0.2, 1],
        'rank': [0, 0.5, 2, 5]
    },
    'cqbg': {
        'id': 12,
        'level': [0, 1, 2, 3, 4],
        'percent': [0, 0.15, 0.25, 0.5, 1],
        'rank': [0, 1, 3, 5, 10]
    },
    'ywchr': {
        'id': 13,
        'level': [0, 1, 2, 3],
        'percent': [0, 0.1, 0.3, 1],
        'rank': [0, 1, 2, 5]
    },
    'shl': {
        'id': 13,
        'level': [0, 1, 2, 3, 4],
        'percent': [0, 0.05, 0.15, 0.3, 1],
        'rank': [0, 0.5, 2, 5, 10]
    },
    'fsh': {
        'id': 14,
        'level': [0, 1, 2, 3],
        'percent': [0, 0.1, 0.5, 1],
        'rank': [0, 0.5, 2, 5]
    }
}