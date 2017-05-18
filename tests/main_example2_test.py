import unittest
unittest.TestCase.maxDiff = None # Show full diff without concatenation when assertion fails

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

class MainExample2TestCase(unittest.TestCase):
    """Tests for `main.py`."""

    def setUp(self):
        from main import ProcessCLI
        # Obtain from terminal log when run main.py runs `vars(parser.parse_args())`
        args = {
            'data_set': 2,
            'logistic': False,
            'kfold': True,
            'linear': True,
            'knn': 'scikit',
            'training_features': [
                'num-of-doors', 'curb-weight', 'horsepower', 'city-mpg', 'highway-mpg'
            ],
            'target_feature': 'price',
            'multi_class_features': '',
            'exclude_non_numeric': [
                'make', 'fuel-type', 'aspiration', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'fuel-system'
            ],
            'exclude_non_ordinal': '',
            'exclude_out_of_scope': ['symboling', 'normalized-losses'],
            'cleanse_price_format_features': ['price'],
            'convert_feature_words_to_digits': ['num-of-doors', 'num-of-cylinders'],
            'kmeans': True,
            'kmeans_qty': 5,
            'affiliation_feature': '',
            'kfold_qty': 10,
            'hyper_optim': True,
            'hyper_optim_range': 20,
            'suppress_all_plots': False
        }

        # Suppress all plots
        args["suppress_all_plots"] = True
        self.process_cli = ProcessCLI(**args)
        self.event_mod = self.process_cli.map_cli_args_to_event_config(EVENT)

    def tearDown(self):
        del self.process_cli
        del self.event_mod

    def test_valid_config_responds_with_expected_results(self):

        expected_result = {
            'logistic': None,
            'linear': {
                'pre-hyperparameter_optimisation': {
                    'model_type': 'linear',
                    'rmse': 4276.331113497754
                },
                'post-hyperparameter_optimisation': {
                    'model_type': 'linear',
                    'feature_names': 'num-of-doors__curb-weight',
                    'rmse': 4235.3551812093447,
                    'k_neighbors_qty': 1,
                    'k_folds_qty': 10,
                    'k_fold_cross_validation_toggle': True
                }
            },
            'knn': {
                'feature_names': 'num-of-doors__curb-weight',
                'rmse': 3945.9183360130905,
                'k_neighbors_qty': 6,
                'k_folds_qty': 10,
                'k_fold_cross_validation_toggle': True
            }
        }

        res = self.process_cli.main(self.event_mod, None)

        self.assertEqual(res['linear']['pre-hyperparameter_optimisation'], expected_result['linear']['pre-hyperparameter_optimisation'])
        self.assertAlmostEqual(res['linear']['post-hyperparameter_optimisation']['rmse'], expected_result['linear']['post-hyperparameter_optimisation']['rmse'])
        self.assertEqual(res['knn'], expected_result['knn'])

if __name__ == '__main__':
    unittest.main()
