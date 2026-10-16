# skewed_rectangle_cython.pyx
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def point_in_skewed_rectangle_cython(p, a, b, c, d):
    cdef tuple ap, ab, ad, bp, bc, ba
    cdef double ap_ab, ap_ad, bp_bc, bp_ba

    ap = (p[0] - a[0], p[1] - a[1])
    ab = (b[0] - a[0], b[1] - a[1])
    ad = (d[0] - a[0], d[1] - a[1])

    ap_ab = ap[0] * ab[1] - ap[1] * ab[0]
    ap_ad = ap[0] * ad[1] - ap[1] * ad[0]

    bp = (p[0] - b[0], p[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    ba = (a[0] - b[0], a[1] - b[1])

    bp_bc = bp[0] * bc[1] - bp[1] * bc[0]
    bp_ba = bp[0] * ba[1] - bp[1] * ba[0]

    if ap_ab * ap_ad >= 0 and bp_bc * bp_ba >= 0:
        return True
    else:
        return False
