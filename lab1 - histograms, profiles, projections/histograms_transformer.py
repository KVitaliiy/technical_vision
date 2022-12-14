from os import listdir
from os.path import splitext, join
from pathlib import Path
import numpy as np
from numpy import ndarray
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

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


def stretching_transform(img: ndarray, a: float = 1) -> ndarray:
    i_max, i_min = np.max(img), np.min(img)
    return (255*(np.power((img - i_min)/(i_max - i_min), a))).astype(np.uint8)


def uniform_transform(img: ndarray, cum_hist: ndarray) -> ndarray:
    i_max, i_min = np.max(img), np.min(img)
    new_img = ndarray(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            new_img[x][y] = (i_max - i_min) * cum_hist[img[x][y]] + i_min
    return new_img.astype(np.uint8)


def exponential_transform(img: ndarray, cum_hist: ndarray, a: float = 0.01) -> ndarray:
    i_min = np.min(img)
    new_img = ndarray(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            new_img[x][y] = i_min - 1/a * np.log(1 - cum_hist[img[x][y]])
    return new_img.astype(np.uint8)


def rayleigh_low_transform(img: ndarray, cum_hist: ndarray, a: float = 100):
    i_min = np.min(img)
    new_img = ndarray(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            new_img[x][y] = i_min + np.power(2*np.power(a, 2) * np.log(1 / (1 - cum_hist[img[x][y]])), 0.5)
    return new_img.astype(np.uint8)


def two_thirds_low_transform(img: ndarray, cum_hist: ndarray):
    new_img = ndarray(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            print(cum_hist[img[x][y]])
            new_img[x][y] = 255*np.power((cum_hist[img[x][y]]), 2/3)
    return new_img.astype(np.uint8)


def hyperbolic_transform(img: ndarray, cum_hist: ndarray, a=None):
    if a is None:
        a = np.min(img)
    if a == 0 or a == 1:
        a = 2
    new_img = ndarray(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            print(cum_hist[img[x][y]])
            new_img[x][y] = 255*np.power(a, cum_hist[img[x][y]])
    return new_img.astype(np.uint8)


def create_sabattier_lut():
    lut = np.arange(256, dtype=np.uint8)
    lut = 4*lut*(255-lut)
    lut = np.where(lut > 0, lut, 0)
    lut = np.clip(lut, 0, 255)
    return lut


def projection_y(img):
    return np.sum(img, 0) / 255


def projection_x(img):
    return np.sum(img, 1) / 255


def profile_x(img):
    return img[img.shape[1] // 2]


def profile_y(img):
    return img[round(img.shape[0] / 2), :]


def plot_projection_y(projection, len_y, subplot, title=None, label_y=None):
    padding = 1.1
    subplot.set_title(title)
    subplot.set_xlim([0, int(np.max(projection))*padding])
    subplot.set_ylim([0, len_y])
    subplot.plot(list(projection), list(range(len_y)))
    subplot.set_ylabel(label_y)


def plot_projection_x(projection, len_x, subplot, title=None, label_y=None):
    padding = 1.1
    subplot.set_title(title)
    subplot.set_xlim([0, len_x])
    subplot.set_ylim([0, int(np.max(projection)*padding)])
    subplot.plot((range(len_x)), projection)
    subplot.set_ylabel(label_y)


def plot_histogram(hist: ndarray, subplot, title=None, label_y=None):
    subplot.set_title(title)
    subplot.set_xlim([0, 255])
    subplot.set_ylim([0, int(np.max(hist))])
    subplot.fill_between(list(range(256)), 0, list(map(int, hist)))
    subplot.set_ylabel(label_y)


def plot_cum_histogram(hist: ndarray, subplot, title=None, label_y=None):
    subplot.set_title(title)
    subplot.set_xlim([0, 255])
    subplot.set_ylim([0, int(np.max(hist))])
    subplot.plot(list(range(256)), list(hist))
    subplot.set_ylabel(label_y)


def plot_image(img: ndarray, subplot, title=None):
    subplot.set_title(title)
    subplot.imshow(img, cmap='gray', aspect='auto')
    subplot.axis('off')


def plot_img_projections(img, projection_y, projection_x):
    fig, ((i1, pry), (prx, empty)) = plt.subplots(2, 2, figsize=(5, 4))
    plot_image(img, i1, 'moon')
    plot_projection_x(projection_x, img.shape[1], prx, 'Projection x')
    plot_projection_y(projection_y, img.shape[0], pry, 'Projection y')
    empty.axis('off')
    fig.tight_layout()
    plt.show()


def plot_profile_x(prf: ndarray, subplot, title=''):
    shift = 1.1
    subplot.set_title(title)
    subplot.set_xlim(len(prf))
    subplot.set_ylim(shift*max(prf))
    subplot.plot(list(range(len(prf))), list(prf))
    subplot.invert_yaxis()


def plot_profile_x_image(img, profile):
    fig, (i, pr) = plt.subplots(2, 1, figsize=(10, 7))
    plot_image(img, i, 'barcode')
    plot_profile_x(profile, pr)
    plt.show()


def plot_images_comparison(hist1: ndarray, img1: ndarray, cum_hist1: ndarray,
                           hist2: ndarray, img2: ndarray, cum_hist2: ndarray):
    fig, ((ch1, h1, i1), (ch2, h2, i2)) = plt.subplots(2, 3, figsize=(7, 4.6))
    plot_cum_histogram(cum_hist1, ch1, 'Cumulative histogram', 'original')
    plot_histogram(hist1, h1, 'Histogram')
    plot_image(img1, i1, 'dog')
    plot_cum_histogram(cum_hist2, ch2, label_y='after transformation')
    plot_histogram(hist2, h2)
    plot_image(img2, i2)
    fig.tight_layout()
    plt.show()


def plot_squared_border_image(img: ndarray, point: tuple, side_a, side_b: int):
    plt.imshow(img, cmap='gray', origin='lower')
    plt.gca().add_patch(Rectangle(point, side_a, side_b, linewidth=1, edgecolor='r', fill=False))
    plt.axis('off')
    plt.savefig('moon_with_borders.png', bbox_inches='tight', pad_inches=0)


def cum_histogram(hist, num_rows, num_column):
    return np.cumsum(hist) / (num_rows * num_column)


if __name__ == '__main__':
    pass
