import sys
import unittest
sys.path.append("../src")  # This is relative to test directory e.g. /realestatereview/test
from data import read_data_from_file

# @unittest.skip("Class disabled")
class TestData(unittest.TestCase):
    def setUp(self):
        ''' this is called once before each test method '''
        self.train_data, self.train_images =  read_data_from_file('synimg/train/data.csv', max_per_class=100)


    def test_read_data(self):
        self.assertEqual(len(self.train_data), 100*10)