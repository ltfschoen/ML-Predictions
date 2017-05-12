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

class PredictionConfigTestCase(unittest.TestCase):
    """Tests for `prediction_config.py`."""

    def setUp(self):
        self.prediction_config = PredictionConfig(EVENT, None)

    def tearDown(self):
        del self.prediction_config

    def test_valid_config_toggles_kfold_cross_validation_off_when_using_manual_knn_model(self):

        # Setup
        self.prediction_config.ML_MODEL_KNN = "manual"
        self.prediction_config.K_FOLD_CROSS_VALIDATION = True

        # Test
        self.prediction_config.validate_config()
        self.assertEqual(self.prediction_config.K_FOLD_CROSS_VALIDATION, False)

    def test_valid_config_toggles_hyperparameter_optimisation_off_when_using_kfold_cross_validation(self):

        # Setup
        self.prediction_config.K_FOLD_CROSS_VALIDATION = True
        self.prediction_config.HYPERPARAMETER_OPTIMISATION = False

        # Test
        self.prediction_config.validate_config()
        self.assertEqual(self.prediction_config.HYPERPARAMETER_OPTIMISATION, True)

if __name__ == '__main__':
    unittest.main()
