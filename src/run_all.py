from data import read_data_from_file, read_labels, write_output, get_labels
from features import extract_features, convert_nearest_std_colors
from model import model_selection, dummy_model
import numpy as np
import warnings
# warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # Read input
    MAX_PER_CLASS = 500
    MAX_TEST_ROWS = None
    train_data, train_images =  read_data_from_file('synimg/train/data.csv', max_per_class=MAX_PER_CLASS)
    test_data, test_images = read_data_from_file('synimg/test/data_nostyle.csv', nrows = MAX_TEST_ROWS)
    label_encoder, train_data = get_labels(train_data, print_classes=False) # one-hot encode, returns in column 'style_id'
    print('test_data', test_data.shape)

    # Preprocessing features
    X_train, X_test = extract_features(train_images, cachefile="train_{}".format(MAX_PER_CLASS)),\
        extract_features(test_images, cachefile="test".format(MAX_TEST_ROWS))
    y_train = list(train_data['style_id'])

    # Train / select model
    model = model_selection(X_train, y_train)  # return a trained model that implements predict()from a bunch of candidates model

    # Predict
    test_data['label_id'] = model.predict(X_test)

    # Submit
    write_output(test_data, label_encoder)