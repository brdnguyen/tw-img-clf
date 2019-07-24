import scipy.spatial as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

main_colors = [
    (0,0,0), (255,255,255), (255,0,0), (0,255,0),
    (0,0,255), (255,255,0), (0,255,255), (255,0,255)
]


def rescale(features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(features)
    return rescaled_features


def extract_features(images, cachefile=None):
    print("Extract features starting...")
    if cachefile and os.path.exists(cachefile + ".npz"):
        features = np.load(cachefile + ".npz")['arr_0']
        print('loaded from cache')
        return features
    print("process from scratch")
    features = [image_histogram(image) for image in images]
    if cachefile:
        np.savez(cachefile, features)
    return features


def image_histogram(image):
    ''' return histogram of the main colors '''
    image, hist = convert_nearest_std_colors(image)
    return hist


def convert_nearest_std_colors(image):
    ''' convert each pixel image to its nearest color, sort of Gaussian filtering,
    but do it explicitly '''
    h, w, bpp = np.shape(image)
    hist = np.zeros(len(main_colors))
    old_image = image.copy()

    for py in range(0,h):
        for px in range(0,w):
            input_color = (image[py][px][0],image[py][px][1],image[py][px][2])
            tree = sp.KDTree(main_colors)
            distance, nearest_idx = tree.query(input_color)
            nearest_color = main_colors[nearest_idx]
            image[py][px] = [nearest_color[0], nearest_color[1], nearest_color[2]]
            hist[nearest_idx] += 1

    # show image
    # f, axarr = plt.subplots(1,2)
    # axarr[0].imshow(old_image)
    # axarr[1].imshow(image)
    # plt.show()
    return image, hist