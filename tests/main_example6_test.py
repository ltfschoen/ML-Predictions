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

class MainExample4NonLogisticTestCase(unittest.TestCase):
    """Tests for `main.py`."""

    def setUp(self):
        from main import ProcessCLI
        # Obtain from terminal log when run main.py runs `vars(parser.parse_args())`
        args = {
            'data_set': 6,
            'logistic': False,
            'kfold': True,
            'linear': True,
            'knn': 'scikit',
            'training_features': [
                'yearpublished',
                'minplaytime',
                'minage',
                'total_wanters',
                'average_weight'
            ],
            'target_feature': 'average_rating',
            'multi_class_features': '',
            'exclude_non_numeric': [
                'type,',
                'name'
            ],
            'exclude_non_ordinal': '',
            'exclude_out_of_scope': [
                'id'
            ],
            'cleanse_price_format_features': '',
            'convert_feature_words_to_digits': '',
            'kmeans': True,
            'kmeans_qty': 5,
            'affiliation_feature': 'yearpublished',
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
                    'rmse': 1.2042523903063078
                },
                'post-hyperparameter_optimisation': {
                    'model_type': 'linear',
                    'feature_names': 'yearpublished__minage__total_wanters__average_weight',
                    'rmse': 1.2096966125815165,
                    'k_neighbors_qty': 1,
                    'k_folds_qty': 10,
                    'k_fold_cross_validation_toggle': True
                }
            },
            'knn': {
                'feature_names': 'yearpublished__minplaytime__minage__total_wanters__average_weight',
                'rmse': 1.1749918955167673,
                'k_neighbors_qty': 15,
                'k_folds_qty': 10,
                'k_fold_cross_validation_toggle': True
            }
        }

        self.assertEqual(self.process_cli.main(self.event_mod, None), expected_result)

if __name__ == '__main__':
    unittest.main()
