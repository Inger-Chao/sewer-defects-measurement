
from pipe_calibration import ShowVideos, ShowDatasets, ShowImages
from data_loader import videos_path, load_songbai_data
from config import conf
import numpy as np

from utils.utils import printResults

# for video in videos_path():
#     ShowVideos(video)


match, acc = ShowDatasets(conf.get("datasets"))
printResults(acc)
# print(match)
# print(acc)
# ShowImages(load_songbai_data())

# all= np.array([[36,230,36,14,33,431,33,6,98,5],
# [13,97,14,7,22,39,12,3,10,5],
# [14,25,1,4,12,22,12,3,8,4],
# [16,39,1,1,1,31,15,2,6,5]])

# match = np.array([[35,226,35,14,32,429,31,4,94,5],
# [11,95,10,2,19,26,8,2,6,2],
# [9,24,0,2,5,12,6,1,6,0],
# [7,34,0,0,0,15,9,0,0,2]])

# acc = np.zeros((4,10))

# for i in range(4):
#     for j in range(10):
#         acc[i][j] = round(match[i][j]/all[i][j],3) * 100

# print(acc)
