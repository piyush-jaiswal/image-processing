from os import path, mkdir

import numpy as np
import cv2

from utils import contours, morphology


def run(img, out_dir, write_intermediary=True):
    if not path.exists(out_dir):
        mkdir(out_dir)

    # -------------------------------------------------# Step 1---------------------------------------------------------
    h, w, _ = img.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    floodflags = 4
    floodflags |= (255 << 8)
    floodflags |= cv2.FLOODFILL_MASK_ONLY

    seed = (0, 0)
    low = 2
    high = 2
    _, _, mask, _ = cv2.floodFill(img, mask, seed, (255, 0, 0), (low,) * 3, (high,) * 3, floodflags)
    mask = 255 - mask

    if write_intermediary:
        cv2.imwrite(path.join(out_dir, '1.jpg'), mask)

    # -------------------------------------------------# Step 2---------------------------------------------------------
    kernel = morphology.get_kernel(30)
    mask_opened = morphology.opening(mask, kernel)

    if write_intermediary:
        cv2.imwrite(path.join(out_dir, '2.jpg'), mask_opened)

    # -------------------------------------------------# Step 3---------------------------------------------------------
    mask_opened_closed = morphology.closing(mask_opened, kernel)

    if write_intermediary:
        cv2.imwrite(path.join(out_dir, '3.jpg'), mask_opened_closed)

    # -------------------------------------------------# Step 4---------------------------------------------------------
    cnts = contours.get_contours(mask_opened_closed)
    hulls = contours.get_convex_hull(cnts)
    img_hull = contours.write_contours(hulls, mask_opened_closed.shape)

    if write_intermediary:
        cv2.imwrite(path.join(out_dir, '4.jpg'), img_hull)

    # -------------------------------------------------# Step 5---------------------------------------------------------
    floodflags = 4
    floodflags |= (255 << 8)
    low = 100
    high = 100
    _, _, mask, _ = cv2.floodFill(img, img_hull, seed, (0, 0, 0), (low,) * 3, (high,) * 3, floodflags)
    cv2.imwrite(path.join(out_dir, 'extacted.jpg'), img)


if __name__ == '__main__':
    img_path = r'input/diamond1.jpg'
    img = cv2.imread(img_path, 1)
    out_dir = r'output'
    run(img, out_dir)
