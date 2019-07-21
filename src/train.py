from data import read_data_from_file, read_labels, write_output, get_labels
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
SEED = 15


def dummy_model(images, NCLASS):
    ''' Dummy model for now '''
    import random
    preds = []
    for idx, img in enumerate(images):
        preds.append(random.randint(0, NCLASS - 1))
    return preds


def fit_predict(model, X, y, X_cv):
    model.fit(X, y)
    preds = model.predict(X_cv)
    return preds


def cal_score(y_preds, y):
    print("y_preds", y_preds)
    print("y", y)
    l = sum([1 if y_preds[idx] == y[idx] else 0 for idx in range(len(y_preds))])
    return l / len(y_preds) * 1.0


def extract_features(images):
    return [image.flatten() for image in images]


def cross_validate(model, X_train, y_train):
    KFOLD = 3 #3 is enough to tell the range of scoring
    mean_auc = 0.
    for i in range(KFOLD):
        print("calculating fold ", i + 1)
        X, X_cv, y, y_cv = train_test_split(
            X_train, y_train,
            test_size = 0.2,
            random_state = i*SEED
        )

        y_preds = fit_predict(model, X, y, X_cv)
        scoring = cal_score(y_preds, y_cv)

        print("Score (fold %d/%d): %f" % (i + 1, KFOLD, scoring))
        mean_auc += scoring
    return mean_auc/KFOLD

if __name__ == '__main__':
    # Read input
    train_data, train_images =  read_data_from_file('synimg/train/data.csv', max_per_class=1000)
    test_data, test_images = read_data_from_file('synimg/test/data_nostyle.csv', nrows=100)
    label_encoder, train_data = get_labels(train_data, print_classes=True) # one-hot encode, returns in column 'style_id'

    # Preprocessing features
    X_train, X_test = extract_features(train_images), extract_features(test_images)
    y_train = list(train_data['style_id'])

    # Train model
    model = LogisticRegression(solver='lbfgs', multi_class="auto")
    model = cross_validate(model, X_train, y_train)  # check model performance

    # Predict
    a = dummy_model(test_images, len(label_encoder.classes_))
    test_data['label_id'] =a

    # Submit
    write_output(test_data, label_encoder)