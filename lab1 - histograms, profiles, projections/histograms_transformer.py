from os import listdir
from os.path import splitext, join
from pathlib import Path
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from math import pow
import matplotlib.cm as cm


EXTENSIONS_FOTO = [".jpg", ".jfif", ".png"]
PATH_TO_FOTO = "images"


def get_image_with_min_contrast(path_to_dir: Path):
    min_contrast = 256
    result_img = None
    for file in listdir(path_to_dir):
        extension = splitext(file)[1]
        if extension in EXTENSIONS_FOTO:
            file = join(path_to_dir, file)
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)
            contrast = np.max(img) - np.min(img)
            if contrast < min_contrast:
                min_contrast = contrast
                result_img = img
    return result_img


def linear_transform(img, shift=0):
    return img + shift


def plot_hists_images(hist1, img1, hist2, img2):
    fig, ((h1, i1), (h2, i2)) = plt.subplots(2, 2, figsize=(5, 4))
    h1.set_title('Histogram')
    h1.set_xlim([0, 255])
    h1.set_ylim([0, int(np.max(hist1))])
    h1.fill_between(list(range(256)), 0, list(map(int, hist1)))
    h1.set_ylabel('original')
    i1.set_title('Castle')
    i1.imshow(img1, cmap='gray', vmin=0, vmax=255, aspect='auto')
    i1.axis('off')
    h2.set_xlim([0, 255])
    h2.set_ylim([0, int(np.max(hist2))])
    h2.set_ylabel('after conversion')
    h2.fill_between(list(range(256)), 0, list(map(int, hist2)))
    i2.imshow(img2, cmap='gray', vmin=0, vmax=255, aspect='auto')
    i2.axis('off')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    image = get_image_with_min_contrast(Path(PATH_TO_FOTO))
    histogram = cv.calcHist([image], [0], None, [256], [0, 256])
    image_2 = linear_transform(image, 2)
    # histogram_2 = cv.calcHist([image_2], [0], None, [256], [0, 256])
    # plot_hists_images(histogram, image, histogram_2, image_2)

