import numpy as np
import cv2 as cv

from math import cos, sin
from numpy.random import random_sample

def bar2cart(polygon, alphas):
    return np.sum(polygon * alphas, axis=0)

def get_centroid(contour):
    M = cv.moments(contour)

    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    return np.array([cx, cy])

def normalize_contour(contour):
    return contour - get_centroid(contour)

def warp_contour(cnt, translation, angle):
    rot_mat = np.array([[cos(angle), -sin(angle)],
                        [sin(angle), cos(angle)]])

    mul = rot_mat[np.newaxis, ...] @ cnt[..., np.newaxis]
    rotated = mul.reshape(-1, 2)
    translated = rotated + translation
    return translated.astype(np.int32)

def random_in_range(low: float, high: float, size=None):
    return (high - low) * random_sample(size) + low