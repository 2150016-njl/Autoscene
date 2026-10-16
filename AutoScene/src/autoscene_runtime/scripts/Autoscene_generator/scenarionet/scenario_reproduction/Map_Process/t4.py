# import time
# import numpy as np
# from numba import jit
#
#
# def point_in_skewed_rectangle(p, a, b, c, d):
#     ap = (p[0] - a[0], p[1] - a[1])
#     ab = (b[0] - a[0], b[1] - a[1])
#     ad = (d[0] - a[0], d[1] - a[1])
#
#     ap_ab = ap[0] * ab[1] - ap[1] * ab[0]
#     ap_ad = ap[0] * ad[1] - ap[1] * ad[0]
#
#     bp = (p[0] - b[0], p[1] - b[1])
#     bc = (c[0] - b[0], c[1] - b[1])
#     ba = (a[0] - b[0], a[1] - b[1])
#
#     bp_bc = bp[0] * bc[1] - bp[1] * bc[0]
#     bp_ba = bp[0] * ba[1] - bp[1] * ba[0]
#
#     return ap_ab * ap_ad >= 0 and bp_bc * bp_ba >= 0
#
# if __name__ == '__main__':
#     # 测试示例
#     p = np.array((1, 1))
#     a = np.array((0, 0))
#     b = np.array((4, 0))
#     c = np.array((3, 3))
#     d = np.array((-1, 3))
#
#     t1 = time.time()
#     for i in range(66618):
#         result = point_in_skewed_rectangle(p, a, b, c, d)
#     t2 = time.time()
#     print(t2 - t1)

import time
from skewed_rectangle_cython import point_in_skewed_rectangle_cython

if __name__ == '__main__':
    p = (1, 1)
    a = (0, 0)
    b = (4, 0)
    c = (3, 3)
    d = (-1, 3)

    t1 = time.time()
    for i in range(66618):
        result = point_in_skewed_rectangle_cython(p, a, b, c, d)
    t2 = time.time()
    print(t2 - t1)
