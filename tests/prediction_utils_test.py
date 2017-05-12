import unittest

import sys
import site

def get_main_path():
    test_path = sys.path[0] # sys.path[0] is current path in lib subdirectory
    split_on_char = "/"
    return split_on_char.join(test_path.split(split_on_char)[:-1])
main_path = get_main_path()
site.addsitedir(main_path+'/tests')
site.addsitedir(main_path+'/lib')
print ("Imported subfolder: %s" % (main_path+'/tests') )

from tests.input_event_test import EVENT
from prediction_config import PredictionConfig
from prediction_utils import PredictionUtils
from prediction_data import PredictionData

class PredictionDataTestCase(unittest.TestCase):
    """Tests for `prediction_utils.py`."""

    def setUp(self):
        self.prediction_config = PredictionConfig(EVENT, None)
        self.prediction_utils = PredictionUtils(self.prediction_config)
        self.prediction_data = PredictionData(self.prediction_config, self.prediction_utils)
        self.dataset_choice = self.prediction_config.DATASET_CHOICE

    def tearDown(self):
        del self.prediction_config
        del self.prediction_utils
        del self.prediction_data
        del self.dataset_choice

    def test_calc_sensitivity(self):

        # Setup
        _count_true_positives = 4
        _count_false_negatives = 6
        _expected_sensitivity = 0.4

        # Test
        self.assertEqual(self.prediction_utils.calc_sensitivity(_count_true_positives, _count_false_negatives), _expected_sensitivity)

    def test_calc_specificity(self):

        # Setup
        _count_true_negatives = 4
        _count_false_positives = 6
        _expected_sensitivity = 0.4

        # Test
        self.assertEqual(self.prediction_utils.calc_sensitivity(_count_true_negatives, _count_false_positives), _expected_sensitivity)


if __name__ == '__main__':
    unittest.main()
