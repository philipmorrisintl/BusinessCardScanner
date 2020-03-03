# Copyright Philip Morris Products S.A. 2019

import cv2
import base64
import numpy as np
from scipy import stats


class DebugUtils:

    def __init__(self, debug=False):
        self.debug = debug

    def set_debug(self, debug=True):
        self.debug = debug

    def log(self, *args):
        if not self.debug:
            return
        print(*args)

    def imshow(self, win_name, image, display_height=750):

        if not self.debug:
            return
        scale = display_height / image.shape[0]
        display_image = cv2.resize(image, None, fx=scale, fy=scale)
        cv2.imshow(win_name, display_image)

    def imcopy(self, image):

        if not self.debug:
            return None
        return image.copy()

    def draw_box(self, image, box, color=(0, 0, 255), thickness=2):

        if not self.debug:
            return
        for j in range(4):
            p1 = box[j]
            p2 = box[(j + 1) % 4]
            cv2.line(image, (p1[0], p1[1]), (p2[0], p2[1]), color, thickness, cv2.LINE_AA)

    def draw_contours(self, image, contours, color=(0, 255, 255), thickness=2):

        if not self.debug:
            return

        cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def as_data_url(image, extension="png"):

    if image is None:
        return None

    imtype = extension.rstrip(".")
    imext = "." + imtype
    _, image_bytes = cv2.imencode(imext, image)
    image_base64 = base64.b64encode(image_bytes).decode()
    return 'data:image/{};base64,{}'.format(imtype, image_base64)


def pad_and_resize(image, target_width, target_height):

    height = image.shape[0]
    width = image.shape[1]

    ar = width / float(height)
    target_ar = target_width / float(target_height)

    padded_width = int(width if ar > target_ar else height * target_ar)
    padded_height = int(height if ar < target_ar else width / target_ar)

    padded_image = np.zeros((padded_width, padded_height, 3), np.uint8)
    padded_image[:height, :width] = image
    resized_image = cv2.resize(padded_image, (target_width, target_height))

    r_w = padded_width / float(target_width)
    r_h = padded_height / float(target_height)

    return r_w, r_h, resized_image


def remove_outliers(X, sigmas=5):

    median = np.median(X, axis=0)
    mad = stats.median_absolute_deviation(X)
    if mad == 0:
        return np.arange(0, len(X), 1).astype(np.int32)
    z_score = 0.6745 * np.sqrt((X - median) ** 2) / mad

    inliers = np.where(z_score < sigmas)[0]

    return inliers


def sort_box(box):

    box = np.array(box).astype("float")

    # Find the Center of Mass: data is a numpy array of shape (Npoints, 2)
    mean = np.mean(box, axis=0)
    # Compute angles
    angles = np.arctan2((box - mean)[:, 1], (box - mean)[:, 0])
    # Transform angles from [-pi,pi] -> [-3*pi/2, pi/2]
    angles -= np.pi / 2

    # Sort
    sorting_indices = np.argsort(angles)
    sorted_box = box[sorting_indices]

    return sorted_box
