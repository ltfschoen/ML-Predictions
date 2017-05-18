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
            'data_set': 4,
            'logistic': True,
            'kfold': False,
            'linear': True,
            'knn': 'scikit',
            'training_features': [
                'gpa',
                'gre'
            ],
            'target_feature': 'admit',
            'multi_class_features': '',
            'exclude_non_numeric': '',
            'exclude_non_ordinal': '',
            'exclude_out_of_scope': '',
            'cleanse_price_format_features': '',
            'convert_feature_words_to_digits': '',
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
            'logistic': {
                'pre-hyperparameter_optimisation': {
                    'model_type': 'logistic',
                    'rmse': 0.393589629343064,
                    'accuracy': 0.782608695652174,
                    'sensitivity': 0.5,
                    'specificity': 0.5,
                    'auc_score': 0.85679303278688534
                }
            },
            'linear': {
                'pre-hyperparameter_optimisation': {
                    'model_type': 'linear',
                    'rmse': 0.39200813287845615
                }
            },
            'knn': {
                'feature_names': 'gpa__gre',
                'rmse': 0.6196196990320526,
                'k_neighbors_qty': 8,
                'k_folds_qty': 10,
                'k_fold_cross_validation_toggle': False
            }
        }

        self.assertEqual(self.process_cli.main(self.event_mod, None), expected_result)

if __name__ == '__main__':
    unittest.main()
