"""Geometry-only interaction checks. No ROS, model or vehicle control side effects."""
import math


def boxes_overlap(a, b):
    """SAT on (x, y, yaw, length, width), including boundary contact."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    au = (math.cos(a[2]), math.sin(a[2]))
    av = (-au[1], au[0])
    bu = (math.cos(b[2]), math.sin(b[2]))
    bv = (-bu[1], bu[0])
    for axis in (au, av, bu, bv):
        dot = lambda u: abs(u[0] * axis[0] + u[1] * axis[1])
        radius = a[3] * .5 * dot(au) + a[4] * .5 * dot(av)
        radius += b[3] * .5 * dot(bu) + b[4] * .5 * dot(bv)
        if abs(dx * axis[0] + dy * axis[1]) > radius + 1e-9:
            return False
    return True


def collision_pairs(current, previous=None):
    """All vehicle pairs, with substeps to detect between-frame crossing.

    Bodies are {stable_id: (x,y,yaw,length,width)}. Interpolation checks relative
    translation at <= 0.25 m and rotation at <= 0.05 rad. It is not a rigid-body
    collision solver; physics remains the vehicle engine's responsibility.
    """
    previous = previous or {}
    pairs = []
    ids = sorted(current)
    for i, left in enumerate(ids):
        a = current[left]
        if not all(math.isfinite(float(v)) for v in a) or min(a[3:]) <= 0:
            raise ValueError('Invalid collision body: ' + left)
        for right in ids[i + 1:]:
            b = current[right]
            if not all(math.isfinite(float(v)) for v in b) or min(b[3:]) <= 0:
                raise ValueError('Invalid collision body: ' + right)
            ap, bp = previous.get(left, a), previous.get(right, b)
            radius = .5 * (math.hypot(a[3], a[4]) + math.hypot(b[3], b[4]))
            if (max(ap[0], a[0]) + radius < min(bp[0], b[0]) or
                    max(bp[0], b[0]) + radius < min(ap[0], a[0]) or
                    max(ap[1], a[1]) + radius < min(bp[1], b[1]) or
                    max(bp[1], b[1]) + radius < min(ap[1], a[1])):
                continue
            da = (a[2] - ap[2] + math.pi) % (2 * math.pi) - math.pi
            db = (b[2] - bp[2] + math.pi) % (2 * math.pi) - math.pi
            travel = math.hypot((a[0]-ap[0])-(b[0]-bp[0]), (a[1]-ap[1])-(b[1]-bp[1]))
            steps = max(1, int(math.ceil(travel / .25)), int(math.ceil(max(abs(da), abs(db)) / .05)))
            for step in range(steps + 1):
                f = step / steps
                aa = (ap[0]+f*(a[0]-ap[0]), ap[1]+f*(a[1]-ap[1]), ap[2]+f*da, a[3], a[4])
                bb = (bp[0]+f*(b[0]-bp[0]), bp[1]+f*(b[1]-bp[1]), bp[2]+f*db, b[3], b[4])
                if boxes_overlap(aa, bb):
                    pairs.append([left, right])
                    break
    return pairs
