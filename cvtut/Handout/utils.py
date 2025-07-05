import cv2 as cv
import numpy as np
from typing import List, Tuple
import yaml
import math
import matplotlib.pyplot as plt

def add_to_class(Class):
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

def pointDistance(l: np.ndarray, r: np.ndarray) -> float:
    return np.sqrt((l[0] - r[0]) ** 2 + (l[1] - r[1]) ** 2)

def drawAllContours(img: np.ndarray, light_contours: List[List[np.ndarray]]) -> None:
    img = img.copy()
    for i in range(len(light_contours)):
        cv.drawContours(img, light_contours, i, (0, 255, 0), 2)
    return img

def showImageOnNotebook(img: np.ndarray):
    plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

def drawArmors(img: np.ndarray, armors: List):
    img = img.copy()
    for armor in armors:
        pt = armor.points
        cv.line(img, (int(pt[0][0]), int(pt[0][1])), (int(pt[1][0]), int(pt[1][1])), (0, 255, 0), 3)
        cv.line(img, (int(pt[1][0]), int(pt[1][1])), (int(pt[2][0]), int(pt[2][1])), (0, 255, 0), 3)
        cv.line(img, (int(pt[2][0]), int(pt[2][1])), (int(pt[3][0]), int(pt[3][1])), (0, 255, 0), 3)
        cv.line(img, (int(pt[3][0]), int(pt[3][1])), (int(pt[0][0]), int(pt[0][1])), (0, 255, 0), 3)
        # cv.polylines(img, np.array(armor.points).astype(np.int32).reshape((-1, 1, 2)), True, (0, 255, 0), 3)
        # I don't know why it doesn't work
    return img

class RotatedRect:
    def __init__(self, center: Tuple[float, float], size: Tuple[float, float], angle: float):
        self.center = center
        self.size = size
        self.angle = angle

    def points(self) -> List[Tuple[float, float]]:
        _angle = self.angle * math.pi / 180.0
        b = math.cos(_angle) * 0.5
        a = math.sin(_angle) * 0.5

        pt = [(0.0, 0.0)] * 4
        pt[0] = (self.center[0] - a * self.size[1] - b * self.size[0], self.center[1] + b * self.size[1] - a * self.size[0])
        pt[1] = (self.center[0] + a * self.size[1] - b * self.size[0], self.center[1] - b * self.size[1] - a * self.size[0])
        pt[2] = (2 * self.center[0] - pt[0][0], 2 * self.center[1] - pt[0][1])
        pt[3] = (2 * self.center[0] - pt[1][0], 2 * self.center[1] - pt[1][1])

        return pt