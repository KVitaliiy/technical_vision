from os import listdir
from os.path import splitext, join
from pathlib import Path
import numpy as np
from numpy import ndarray
import cv2 as cv
from matplotlib import pyplot as plt


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
    i_max = np.max(img)
    i_min = np.min(img)
    return (255*(np.power((img - i_min)/(i_max - i_min), a))).astype(np.uint8)

# def even_transformation(img : ndarray, cum_hist: ndarray):


def plot_histogram(hist: ndarray, subplot, title=None, label_y=None):
    subplot.set_title(title)
    subplot.set_xlim([0, 255])
    subplot.set_ylim([0, int(np.max(hist))])
    subplot.fill_between(list(range(256)), 0, list(map(int, hist)))
    subplot.set_ylabel(label_y)


def plot_cum_histogram(hist: ndarray, subplot, title=None, label_y=None):
    print(hist.shape)
    print(hist)
    subplot.set_title(title)
    subplot.set_xlim([0, 255])
    subplot.set_ylim([0, int(np.max(hist))])
    subplot.plot(list(range(256)), list(map(int, hist)))
    subplot.set_ylabel(label_y)


def plot_image(img: ndarray, subplot, title=None):
    subplot.set_title(title)
    subplot.imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto')
    subplot.axis('off')


def plot_images_comparison(hist1: ndarray, img1: ndarray, cum_hist1: ndarray,
                           hist2: ndarray, img2: ndarray, cum_hist2: ndarray):
    fig, ((ch1, h1, i1), (ch2, h2, i2)) = plt.subplots(2, 3, figsize=(7, 4.6))
    plot_cum_histogram(cum_hist1, ch1, 'Cumulative histogram', 'original')
    plot_histogram(hist1, h1, 'Histogram')
    plot_image(img1, i1, 'Castle')
    plot_cum_histogram(cum_hist2, ch2, label_y='after transformation')
    plot_histogram(hist2, h2)
    plot_image(img2, i2)
    fig.tight_layout()
    plt.show()


def cum_histogram(hist, num_rows, num_column):
    return 500*np.cumsum(hist) / (num_rows * num_column)


if __name__ == "__main__":
    image = get_image_with_min_contrast(Path(PATH_TO_PHOTO))
    histogram = cv.calcHist([image], [0], None, [256], [0, 256])
    image_2 = stretching(image, 1.9)
    cum_h = cum_histogram(histogram, image.shape[0], image.shape[1])
    histogram_2 = cv.calcHist([image_2], [0], None, [256], [0, 256])
    cumh_2 = cum_histogram(histogram_2, image.shape[0], image.shape[1])
    plot_images_comparison(histogram, image, cum_h, histogram_2, image_2, cumh_2)
