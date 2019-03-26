import numpy as np
from imageio import imread, imsave
import cv2


def dilate(img, kernel, iterations=1):
    return cv2.dilate(img, kernel, iterations)


def erosion(img, kernel, iterations=1):
    return cv2.erode(img, kernel, iterations)


def opening(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def closing(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def morphological_gradient(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


def tophat(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)


def blackhat(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


def get_kernel(kernel_size=3, kernel_type='default'):
    if kernel_type.lower() == 'rectangle':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_type.lower() == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_type.lower() == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    elif kernel_type.lower() == 'default':
        kernel = np.ones((kernel_size, kernel_size))
    else:
        raise ValueError('kernel_type {} not found'.format(kernel_type))

    return kernel


if __name__ == '__main__':
    img_path = r''
    img_out_path = r''

    img = imread(img_path)
    kernel = get_kernel(30)
    opened = opening(img, kernel)
    kernel = get_kernel(30, kernel_type='ellipse')
    closed = closing(opened, kernel)
    imsave(img_out_path, closed)
