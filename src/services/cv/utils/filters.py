import cv2
import numpy as np

KERNEL_9 = [[-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]]

KERNEL_5 = [[0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]]


def sharp_filter(src: np.ndarray, *, strong: bool = False) -> np.ndarray:
    kernel = KERNEL_9 if strong else KERNEL_5
    return cv2.filter2D(src, ddepth=-1, kernel=np.array(kernel))


def clahe_filter(src: np.ndarray, *, clip_limit: float = 1) -> np.ndarray:
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels  # noqa: E741
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
