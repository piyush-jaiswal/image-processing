import cv2
from imageio import imread
from matplotlib import pyplot as plt


def plot_histogram_3d(img, separate=False):
    if len(img.shape) != 3:
        raise ValueError('Image should be 3D!')
    if img.shape[2] != 3:
        raise ValueError('Image should have three channels in the last axis')

    for i, col in enumerate(('r', 'g', 'b')):
        if separate:
            plt.figure(i)
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.show()


def plot_histogram_2d(img):
    if len(img.shape) != 3:
        raise ValueError('Image should be 2D!')

    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


if __name__ == '__main__':
    img_path = r''
    img = imread(img_path)
    plot_histogram_3d(img)
