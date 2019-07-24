import os
import numpy as np
import pandas as pd
import skimage.io as io
from sklearn import preprocessing
import collections

# Some constant
WIDTH = 64
HEIGHT = 32
ROOT_PATH = 'input/synimg/' # relative to project folder


def read_data_from_file(filename, nrows=None, max_per_class=None):
    '''
        File is in the format of data.csv or data_nostyle.csv,
        containing location of each image and IDs and/or la
        filename example: 'synimg/test/data_nostyle.csv'
    '''
    data = pd.read_csv(ROOT_PATH + filename, nrows=nrows)
    if max_per_class:
        data = data.groupby('style_name').head(max_per_class).reset_index()
    # print("Shape and review after getting max per group:\n", data.shape, "\n", data.head(20))
    # data = data.sample(frac=1.0) 
    # Read images
    all_images = read_images(data, max_per_class) # includes caching

    return data, all_images


def read_images(data, max_per_class):
    all_images = []
    for idx, row in data.iterrows():
        img = io.imread(ROOT_PATH + row['filepath'])
        all_images.append(img)
    return all_images


def write_output(test_data, label_encoder):
    test_data['style_name'] = labelId_to_label(test_data['label_id'], label_encoder)
    # Doing some validation
    from collections import Counter
    counter = Counter(test_data['style_name'])
    print(counter)
    test_data[['id', 'style_name']].to_csv('../submit.csv', index=False)
    return test_data, counter


def get_labels(train_data, print_classes=False):
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(train_data['style_name'])
    if print_classes: print("classes: ", label_encoder.classes_)
    train_data['style_id'] = label_encoder.transform(train_data['style_name'])
    return label_encoder, train_data

def labelId_to_label(list_ids, label_encoder):
    return label_encoder.inverse_transform(list_ids)
