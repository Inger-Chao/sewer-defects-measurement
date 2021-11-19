# -*- coding: utf-8 -*-
# @Time    : 2020-11-13 18:55
# @Author  : Inger

import os
import os.path as osp
import numpy as np
from random import sample
import cv2
from config import conf, dft_rank_tbl
from utils.utils import isImageFile, removeDStore

'''json'''
def DataLoader():
    path = conf.get("path")
    imgs = []
    pipes = []
    levels = []
    ids = os.listdir(path)
    removeDStore(ids)
    ids.sort(key= lambda x:int(x))
    for label in ids:
        itemdir = osp.join(path, label)
        img = osp.join(itemdir + "/img.png")
        pipe = cv2.imread(itemdir + "/label.png")
        imgs.append(img)
        pipes.append(pipe)
    return imgs, pipes

class LevelSewer10():
    def __init__(self) -> None:
        self.path = conf.get('datasets')
        self.classes = os.listdir(self.path)
        self.tables = np.zeros(15, 5).astype('int')
        '''准确率'''
        self.cmp_tables = np.zeros(15, 5).astype('int')
        for type in self.classes:
            _type_level_dir = osp.join(self.path, type)
            for level in _type_level_dir:
                type_level_files_num = len(os.listdir(osp.join(_type_level_dir, level)))
                self.tables[dft_rank_tbl.get(type).get('id')][int(level)] = type_level_files_num
        

'''songbai'''
def load_songbai_data():
    path = "datasets/songbai"
    sample = []
    files = os.listdir(path)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    for file in files:
        if isImageFile(file):
            sample.append(os.path.join(path, file))
    return sample

def videos_path():
    ret = []
    path = "/Users/inger/projects/PycharmProjects/opencv_demo/videos/videos-10fps/"
    videos = os.listdir(path)
    for filename in videos:
        video = os.path.join(path, filename)
        ret.append(video)
    return ret
