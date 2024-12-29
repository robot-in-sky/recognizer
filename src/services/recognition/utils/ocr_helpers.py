import cv2
import numpy as np

from .shapes import Rectangle


def detect_box(src: np.ndarray, min_height: float) -> Rectangle:
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    x1, y1, x2, y2 = None, None, None, None
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if h > min_height and hierarchy[0][idx][3] != -1:
            x1 = x if (x1 is None) else min(x1, x)
            y1 = y if (y1 is None) else min(y1, y)
            x2 = x + w if (x2 is None) else max(x2, x + w)
            y2 = y + h if (y2 is None) else max(y2, y + h)

    if None in (x1, y1, x2, y2):
        return Rectangle(0, 0, src.shape[1], src.shape[0])

    margin = 1
    x1 -= margin
    y1 -= margin
    x2 += margin
    y2 += margin
    return Rectangle(x1, y1, x2 - x1, y2 - y1)
