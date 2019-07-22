import scipy.spatial as sp
import matplotlib.pyplot as plt
import numpy as np
import os
main_colors = [
    (0,0,0), (255,255,255), (255,0,0), (0,255,0),
    (0,0,255), (255,255,0), (0,255,255), (255,0,255)
]


def extract_features(images, cachefile):
    print("Extract features starting...")
    filename = 'cache/' + cachefile
    if cachefile and os.path.exists(filename + ".npz"):
        features = np.load(filename + ".npz")['arr_0']
        return features
    features = [image_histogram(image) for image in images]
    if cachefile:
        np.savez(filename, features)
    return features


def image_histogram(image):
    ''' return histogram of the main colors '''
    image, hist = convert_nearest_std_colors(image)
    return hist


def convert_nearest_std_colors(image):
    ''' convert each pixel image to its nearest color, sort of Gaussian filtering,
    but do it explicitly '''
    # print("Before conversion")
    # plt.imshow(image)


    h, w, bpp = np.shape(image)
    hist = np.zeros(len(main_colors))

    for py in range(0,h):
        for px in range(0,w):
            input_color = (image[py][px][0],image[py][px][1],image[py][px][2])
            tree = sp.KDTree(main_colors)
            distance, nearest_idx = tree.query(input_color)
            nearest_color = main_colors[nearest_idx]
            image[py][px][0]=nearest_color[0]
            image[py][px][1]=nearest_color[1]
            image[py][px][2]=nearest_color[2]
            hist[nearest_idx] += 1

    # show image
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(image)
    # plt.show()
    return image, hist