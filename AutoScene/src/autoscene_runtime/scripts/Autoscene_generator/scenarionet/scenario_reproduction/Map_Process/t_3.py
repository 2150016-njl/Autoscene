# import time
# from numba import jit
#
# # @jit
# def point_in_rotated_rectangle(x, y, rect):
#     """
#     判断点 (x, y) 是否在斜的矩形内
#
#     参数：
#     x: 点的 x 坐标
#     y: 点的 y 坐标
#     rect: 斜的矩形，表示为四个顶点的坐标，顺序为左上、右上、右下、左下
#
#     返回值：
#     True，如果点在矩形内；False，如果点在矩形外
#     """
#     p1, p2, p3, p4 = rect
#
#     # 检查点是否在四边形的外部任一侧
#     if (y - p1[1]) * (p2[0] - p1[0]) - (x - p1[0]) * (p2[1] - p1[1]) > 0:
#         return False
#     if (y - p2[1]) * (p3[0] - p2[0]) - (x - p2[0]) * (p3[1] - p2[1]) > 0:
#         return False
#     if (y - p3[1]) * (p4[0] - p3[0]) - (x - p3[0]) * (p4[1] - p3[1]) > 0:
#         return False
#     if (y - p4[1]) * (p1[0] - p4[0]) - (x - p4[0]) * (p1[1] - p4[1]) > 0:
#         return False
#
#     return True
#
# # 示例用法
# point = (1, 1)  # 待判断的点
# rectangle = [(0, 0), (2, 0), (2, 2), (0, 2)]  # 一个斜的矩形，表示为四个顶点的坐标
# t1 = time.time()
# for i in range(66618):
#     if point_in_rotated_rectangle(point[0], point[1], rectangle):
#         # print("Point is inside the rectangle")
#         pass
#     else:
#         # print("Point is outside the rectangle")
#         pass
# t2 = time.time()
# print(t2-t1)


import time
from numba import jit


# @jit
def point_in_skewed_rectangle(p, a, b, c, d):
    def cross_product(p1, p2):
        return p1[0] * p2[1] - p1[1] * p2[0]

    ap = (p[0] - a[0], p[1] - a[1])
    ab = (b[0] - a[0], b[1] - a[1])
    ad = (d[0] - a[0], d[1] - a[1])

    ap_ab = cross_product(ap, ab)
    ap_ad = cross_product(ap, ad)

    bp = (p[0] - b[0], p[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    ba = (a[0] - b[0], a[1] - b[1])

    bp_bc = cross_product(bp, bc)
    bp_ba = cross_product(bp, ba)

    if ap_ab * ap_ad >= 0 and bp_bc * bp_ba >= 0:
        return True
    else:
        return False


if __name__ == '__main__':
    # 测试示例
    p = (1, 1)
    a = (0, 0)
    b = (4, 0)
    c = (3, 3)
    d = (-1, 3)

    t1 = time.time()
    for i in range(66618):
        result = point_in_skewed_rectangle(p, a, b, c, d)
    t2 = time.time()
    print(t2 - t1)
    # print("Point is inside the skewed rectangle:", result)
