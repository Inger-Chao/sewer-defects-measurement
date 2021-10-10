# -*- coding: utf-8 -*-
# @Time    : 2020-11-13 18:55
# @Author  : Inger

import os
import os.path as osp
import cv2
from config import conf

path = conf.get("path")

def DataLoader():
    imgs = []
    pipes = []
    for label in os.listdir(path):
        itemdir = osp.join(path, label)
        print(itemdir)
        img = cv2.imread(itemdir + "/img.png")
        pipe = cv2.imread(itemdir + "/label.png")
        imgs.append(img)
        pipes.append(pipe)
    return imgs, pipes

def load_data(path, cache=True):
    label = list()
    