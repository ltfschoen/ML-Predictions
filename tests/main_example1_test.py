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

class MainExample1TestCase(unittest.TestCase):
    """Tests for `main.py`."""

    def setUp(self):
        from main import ProcessCLI
        # Obtain from terminal log when main.py runs `vars(parser.parse_args())`
        args = {
            'data_set': 1,
            'logistic': False,
            'kfold': True,
            'linear': True,
            'knn': 'scikit',
            'training_features': [
                'accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews'
            ],
            'target_feature': 'price', 'multi_class_features': '',
            'exclude_non_numeric': [
                'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
                'host_listings_count', 'host_verifications', 'host_has_profile_pic', 'host_identity_verified',
                'property_type', 'room_type', 'bed_type', 'amenities', 'calendar_updated', 'has_availability',
                'requires_license', 'license', 'instant_bookable', 'cancellation_policy', 'require_guest_profile_picture',
                'require_guest_phone_verification'
            ],
            'exclude_non_ordinal': [
                'latitude', 'longitude', 'zipcode'
            ],
            'exclude_out_of_scope': [
                'id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'summary', 'space', 'description', 'experiences_offered',
                'neighborhood_overview', 'notes', 'transit', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url',
                'host_id', 'host_url', 'host_name', 'host_since', 'host_location', 'host_about', 'host_thumbnail_url',
                'host_picture_url', 'host_neighbourhood', 'street', 'neighbourhood', 'neighbourhood_cleansed',
                'neighbourhood_group_cleansed', 'city', 'state', 'market', 'smart_location', 'country_code',
                'country', 'is_location_exact', 'calendar_last_scraped', 'first_review', 'last_review', 'jurisdiction_names'
            ],
            'cleanse_price_format_features': [
                'price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee', 'extra_people'
            ],
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
            'logistic': None,
            'linear': {
                'pre-hyperparameter_optimisation': {
                    'model_type': 'linear',
                    'rmse': 111.78870076908206
                },
                'post-hyperparameter_optimisation': {
                    'model_type': 'linear',
                    'feature_names': 'accommodates__bedrooms__bathrooms',
                    'rmse': 109.86142964502258,
                    'k_neighbors_qty': 1,
                    'k_folds_qty': 10,
                    'k_fold_cross_validation_toggle': True
                }
            },
            'knn': {
                'feature_names': 'accommodates__bedrooms__bathrooms',
                'rmse': 109.65178843975404,
                'k_neighbors_qty': 20,
                'k_folds_qty': 10,
                'k_fold_cross_validation_toggle': True
            }
        }

        self.assertEqual(self.process_cli.main(self.event_mod, None), expected_result)

if __name__ == '__main__':
    unittest.main()
