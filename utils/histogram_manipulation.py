import numpy as np
import cv2
from imageio import imread, imsave


def apply_clahe(img, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    # RGB
    if len(img.shape) == 3:
        equ = np.dstack([clahe.apply(img[:, :, i]) for i in range(img.shape[2])])
    # Grayscale
    else:
        equ = clahe.apply(img)

    return equ


if __name__ == '__main__':
    img = imread('/home/jaiswalp/Documents/problems/sample/try.jpg')
    clahe = apply_clahe(img)
    imsave('/home/jaiswalp/Documents/problems/sample/try_clahe.jpg', clahe)
