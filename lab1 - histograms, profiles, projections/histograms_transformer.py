from os import listdir
from os.path import splitext, join
from pathlib import Path
import numpy as np
from numpy import ndarray
import cv2 as cv
from matplotlib import pyplot as plt
from math import pow


EXTENSIONS_PHOTO = [".jpg", ".jfif", ".png"]
PATH_TO_PHOTO = "images"


def get_image_with_min_contrast(path_to_dir: Path) -> ndarray:
    min_contrast = 256
    result_img = None
    for file in listdir(path_to_dir):
        extension = splitext(file)[1]
        if extension in EXTENSIONS_PHOTO:
            file = join(path_to_dir, file)
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)
            contrast = np.max(img) - np.min(img)
            if contrast < min_contrast:
                min_contrast = contrast
                result_img = img
    return result_img


def linear_transform(img: ndarray, shift: float = 0) -> ndarray:
    return img + shift


def stretching(img: ndarray, a: float = 1) -> ndarray:
    new_img = ndarray(img.shape)
    i_max = np.max(img)
    i_min = np.min(img)
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            new_img[j][k] = pow((img[j][k] - i_min)/(i_max - i_min), a)
    return denorm_image(new_img)


def plot_histogram(hist: ndarray, subplot, title=None, label_y=None):
    subplot.set_title(title)
    subplot.set_xlim([0, 255])
    subplot.set_ylim([0, int(np.max(hist))])
    subplot.fill_between(list(range(256)), 0, list(map(int, hist)))
    subplot.set_ylabel(label_y)


def plot_image(img: ndarray, subplot, title=None):
    subplot.set_title(title)
    subplot.imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto')
    subplot.axis('off')


def plot_images_comparison(hist1: ndarray, img1, hist2, img2):
    fig, ((h1, i1), (h2, i2)) = plt.subplots(2, 2, figsize=(5, 4))
    plot_histogram(hist1, h1, 'Histogram', 'original')
    plot_image(img1, i1, 'Castle')
    plot_histogram(hist2, h2, label_y='after transformation')
    plot_image(img2, i2)
    fig.tight_layout()
    plt.show()


def denorm_image(img: ndarray):
    return img * 255


if __name__ == "__main__":
    image = get_image_with_min_contrast(Path(PATH_TO_PHOTO))
    histogram = cv.calcHist([image], [0], None, [256], [0, 256])
    image_2 = stretching(image, 0.7)
    histogram_2 = cv.calcHist([image_2], [0], None, [256], [0, 256])
    plot_images_comparison(histogram, image, histogram_2, image_2)

