import sys
import os
path = sys.argv[1]

level_count = [0, 0, 0, 0, 0]
for file in os.listdir(os.path.abspath(path)):
    if file.endswith('.jpg'):
        level = file.split('-')[1].split('.')[0]
        level_count[int(level)] += 1
print(level_count)
print('总文件：', sum(level_count))