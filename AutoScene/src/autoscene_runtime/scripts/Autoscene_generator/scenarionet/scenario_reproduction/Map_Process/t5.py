# import math
# import random
#
# # 生成66618个点的坐标列表
# import time
#
# num_points = 66618
# points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_points)]
#
#
# # 给定点的坐标
# given_point = (5.2, 5.9)
#
# # 计算与给定点的距离，并找到最小距离对应的点
# min_distance = float('inf')
# closest_point = None
#
# t1= time.time()
# for point in points:
#     distance = math.sqrt((given_point[0] - point[0])**2 + (given_point[1] - point[1])**2)
#     if distance < min_distance:
#         min_distance = distance
#         closest_point = point
# t2 = time.time()
# print("t", t2-t1)
# print("给定点:", given_point)
# print("最近的点:", closest_point)
# print("距离:", min_distance)

import math
import random
import time
import numpy as np
from scipy.spatial import KDTree

# 生成66618个随机点的坐标列表
num_points = 66618
points = np.random.rand(num_points, 2) * 100  # 生成在[0, 100)范围内的随机坐标
t1= time.time()
# 构建KD树
kdtree = KDTree(points)

# 给定点的坐标
given_point = np.array([45.5, 5.6])
# 查询最近的点
# t1= time.time()
distance, closest_index = kdtree.query(given_point.reshape(1, -1), k=1)
closest_point = points[closest_index][0]
t2 = time.time()
print("t", t2-t1)
print("给定点:", given_point)
print("最近的点:", closest_point)
print("距离:", distance[0])
