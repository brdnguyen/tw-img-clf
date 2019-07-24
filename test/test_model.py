import sys
import unittest
sys.path.append("../src")  # This is relative to test directory e.g. /realestatereview/test
from data import read_data_from_file, get_labels
from model import model_selection
from features import extract_features
MAX_PER_CLASS = 5

#TODO: better to have separate tests for each class (data, features, model)
# @unittest.skip("Class disabled")
class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Set up data for the whole TestCase
        self.train_data, self.train_images =  read_data_from_file('synimg/train/data.csv', max_per_class=MAX_PER_CLASS)
        self.label_encoder, self.train_data = get_labels(self.train_data, print_classes=False) # one-hot encode, returns in column 'style_id'
        self.X_train = extract_features(self.train_images)
        self.y_train = list(self.train_data['style_id'])


    def test_read_data(self):
        self.assertEqual(len(self.train_data), MAX_PER_CLASS*10)
        self.assertEqual(len(self.label_encoder.classes_), 10)


    def test_extract_features(self):
        # Preprocessing features
        self.assertEqual(len(self.X_train), len(self.y_train))
        self.assertGreaterEqual(len(self.X_train[0]), 8) # minimum 8 features


    def test_model(self):
        model = model_selection(self.X_train, self.y_train)
        preds = model.predict(self.X_train)
        self.assertEqual(len(preds), MAX_PER_CLASS*10) # model has been trained
