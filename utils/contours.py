import numpy as np
from imageio import imread, imsave
import cv2


def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_approx_contours(contours, epsilon=2):
    return [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]


def write_contours(contours, img_shape, pixel_val=255):
    contour_img = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(contour_img, contours, -1, pixel_val)
    return contour_img


def get_convex_hull(contours):
    return [cv2.convexHull(cnt) for cnt in contours]


if __name__ == '__main__':
    img_path = r''
    epsilon = 2
    img_out_path = r''

    img = imread(img_path)
    contours = get_contours(img)
    # contours = get_approx_contours(contours, epsilon)
    contours = get_convex_hull(contours)
    img_contour = write_contours(contours, img.shape)
    imsave(img_out_path, img_contour)
