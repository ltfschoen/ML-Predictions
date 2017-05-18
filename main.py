#!/usr/bin/env python

from __future__ import print_function # Python 2 or 3
from pprint import pprint
import logging
import sys
import subprocess
import argparse
import textwrap
from logging_config import LoggingConfig
from prediction_data import PredictionData
from prediction_config import PredictionConfig
from prediction_utils import PredictionUtils
import prediction_model_knn_manual
import prediction_model_knn_external
import prediction_model_linear_external
import prediction_model_logistic_external
from input_event import EVENT

DATA_SET_MAPPING = {
    1: "rental-property-listings",
    2: "car-listings",
    3: "car-listings-fuel",
    4: "university-admissions",
    5: "senators-vote",
    6: "game-reviews"
}

class ProcessCLI(object):
    def __init__(self, **args):
        self.data_set = args["data_set"]
        self.logistic = args["logistic"]
        self.kfold = args["kfold"]
        self.linear = args["linear"]
        self.knn = args["knn"]
        self.training_features = self.replace_when_empty(args["training_features"])
        self.target_feature = args["target_feature"]
        self.multi_class_features = self.replace_when_empty(args["multi_class_features"])
        self.exclude_non_numeric = self.replace_when_empty(args["exclude_non_numeric"])
        self.exclude_non_ordinal = self.replace_when_empty(args["exclude_non_ordinal"])
        self.exclude_out_of_scope = self.replace_when_empty(args["exclude_out_of_scope"])
        # self.exclude_ranges_for_features = args["exclude_ranges_for_features"]
        self.cleanse_price_format_features = self.replace_when_empty(args["cleanse_price_format_features"])
        self.convert_feature_words_to_digits = self.replace_when_empty(args["convert_feature_words_to_digits"])
        self.kmeans = args["kmeans"]
        self.kmeans_qty = args["kmeans_qty"]
        self.affiliation_feature = args["affiliation_feature"]
        self.kfold_qty = args["kfold_qty"]
        self.hyper_optim = args["hyper_optim"]
        self.hyper_optim_range = args["hyper_optim_range"]
        self.suppress_all_plots = args["suppress_all_plots"]

    def replace_when_empty(self, arg):
        """ CLI list of args are accepted as string type and argparse convert them to array, but when
        no args passed in we want default of empty array, but can only set an empty string to the default,
        so this function is a workaround"""
        return arg if not arg == '' else []

    def map_cli_args_to_event_config(self, event):
        dataset_selected = DATA_SET_MAPPING[self.data_set]
        event["dataset_selected"] = dataset_selected
        event["model_workflow_config"]["model_workflow_for_logistic_regression_algorithm_toggle"] = self.logistic
        event["k_fold_cross_validation_config"]["k_fold_cross_validation_toggle"] = self.kfold
        event["model_workflow_config"]["model_workflow_for_linear_regression_algorithm_toggle"] = self.linear
        event["model_workflow_config"]["model_workflow_for_knn_regression_algorithm_toggle"] = self.knn
        event["dataset_config"][dataset_selected]["training_columns"] = self.training_features
        event["dataset_config"][dataset_selected]["target_column"] = self.target_feature
        event["dataset_config"][dataset_selected]["multi_classification_input_columns"] = self.multi_class_features
        event["dataset_config"][dataset_selected]["exclude_columns"]["non_numeric"] = self.exclude_non_numeric
        event["dataset_config"][dataset_selected]["exclude_columns"]["non_ordinal"] = self.exclude_non_ordinal
        event["dataset_config"][dataset_selected]["exclude_columns"]["out_of_scope"] = self.exclude_out_of_scope
        # event["dataset_config"][dataset_selected]["exclude_columns"]["exclude_ranges_for_features"] = self.exclude_ranges_for_features
        event["dataset_config"][dataset_selected]["cleanse_columns_price_format"] = self.cleanse_price_format_features
        event["dataset_config"][dataset_selected]["convert_columns_words_to_digits"] = self.convert_feature_words_to_digits
        event["k_means_clustering_config"]["k_means_clustering_toggle"] = self.kmeans
        event["k_means_clustering_config"]["centroids_quantity"] = self.kmeans_qty
        event["dataset_config"][dataset_selected]["affiliation_column"] = self.affiliation_feature
        event["k_fold_cross_validation_config"]["k_folds_quantity"]: self.kfold_qty
        event["hyperparameter_optimisation_config"]["hyperparameter_optimisation_toggle"] = self.hyper_optim
        event["hyperparameter_optimisation_config"]["hyperparameter_range"] = self.hyper_optim_range
        event["plot_config"]["suppress_all_plots"] = self.suppress_all_plots
        return event

    def main(self, event, context):
        logging_config = LoggingConfig()
        logging.info('Starting Prediction')
        prediction_config = PredictionConfig(event, context)
        prediction_utils = PredictionUtils(prediction_config)
        prediction_data = PredictionData(prediction_config, prediction_utils)

        try:
            logistic_results = None
            linear_results = None
            knn_results = None

            # Regression Logistic
            if prediction_config.ML_MODEL_LOGISTIC == True:
                logistic_results = prediction_model_logistic_external.run(prediction_config, prediction_data, prediction_utils)
                for key, value in logistic_results.items():
                    print("RMSE Logistic results for %r: %r" % (key, value))

            # Regression Linear
            if prediction_config.ML_MODEL_LINEAR == True:
                linear_results = prediction_model_linear_external.run(prediction_config, prediction_data, prediction_utils)
                for key, value in linear_results.items():
                    print("RMSE Linear results for %r: %r" % (key, value))

            # Regression with KNN
            if prediction_config.ML_MODEL_KNN == "scikit":
                knn_results = prediction_model_knn_external.run(prediction_config, prediction_data, prediction_utils)
            elif prediction_config.ML_MODEL_KNN == "manual":
                knn_results = prediction_model_knn_manual.run(prediction_config, prediction_data, prediction_utils)
            else:
                print("Unknown KNN Regression option")
                knn_results = None
            res = {
                "logistic": logistic_results,
                "linear": linear_results,
                "knn": knn_results
            }
            print("Predictions: ", res)
            return res
        except RuntimeError as e:
            logging.info('Error: No valid KNN Model selected')

        logging.info('Finished Prediction')

    def get_log_stream(self, event, context):
        """ AWS Lambda Context """

        print("Log stream name:", context.log_stream_name)
        print("Log group name:",  context.log_group_name)
        print("Request ID:",context.aws_request_id)
        print("Mem. limits(MB):", context.memory_limit_in_mb)
        print("Time remaining (MS):", context.get_remaining_time_in_millis())

    def prediction_handler(self, event, context=None):
        """ AWS Lambda Handler
    
        Event argument only provided when using AWS Lambda
        """

        if context:
            self.get_log_stream(context)

        prediction = None

        if event:
            prediction = self.main(event, context)
        else:
            print("Error: Missing event object is required argument of handler")

        return prediction

if __name__ == '__main__':

    # Run Unit Tests
    if '--unittest' in sys.argv:
        subprocess.call([sys.executable, '-m', 'unittest', 'discover', '-s', './tests', '-p', '*_test.py'])
        sys.exit()

    # Parse CLI arguments when the script run
    parser = argparse.ArgumentParser(
        prog='Machine Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
             %(prog)s program
            --------------------------------
            Choose value for data-set:
                1: "rental-property-listings",
                2: "car-listings",
                3: "car-listings-fuel",
                4: "university-admissions",
                5: "senators-vote",
                6: "game-reviews"
            ''')
    )

    required = True
    if '--unittest' in sys.argv:
        required = False

    # Reference: https://docs.python.org/3/library/argparse.html
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')

    parser.add_argument('-ds', '--data-set', type=int, choices=range(1, len(DATA_SET_MAPPING)+1), default=1, help='Data Set Selection')

    logistic_mutex_group = parser.add_mutually_exclusive_group(required=required)
    logistic_mutex_group.add_argument('-log', '--logistic', action='store_true', default=False, help='Logistic Regression Model Toggle')
    logistic_mutex_group.add_argument('-kfc', '--kfold', action='store_true', default=False, help='K-Fold Cross Validation Toggle')

    parser.add_argument('-lin', '--linear', action='store_true', default=True, help='Linear Regression Model Toggle ')
    parser.add_argument('-knn', '--knn', type=str, choices=['manual', 'scikit'], default='scikit', help='KNN Regression Model Toggle')
    parser.add_argument('-trf', '--training-features', type=str, nargs='*', required=required, help='Training Features List')
    parser.add_argument('-taf', '--target-feature', type=str, required=required, help='Target Features')
    parser.add_argument('-mcf', '--multi-class-features', type=str, nargs='*', default='', help='Multi-Classification Features List')

    parser.add_argument('-excn', '--exclude-non-numeric', type=str, nargs='*', default='', help='Exclude Non-Numeric Features List')
    parser.add_argument('-exco', '--exclude-non-ordinal', type=str, nargs='*', default='', help='Exclude Non-Ordinal Features List')
    parser.add_argument('-excs', '--exclude-out-of-scope', type=str, nargs='*', default='', help='Exclude Out-Of-Scope Features List')
    # parser.add_argument('-excr', '--exclude-ranges-for-features', type=str, nargs='*', default='', help='Exclude Ranges for Features List')
    parser.add_argument('-clep', '--cleanse-price-format-features', type=str, nargs='*', default='', help='Cleanse Price Format Features List')
    parser.add_argument('-conwd', '--convert-feature-words-to-digits', type=str, nargs='*', default='', help='Convert Features with Words to Digits List')

    parser.add_argument('-kmc', '--kmeans', action='store_true', default=True, help='K-Means Clustering Toggle')
    parser.add_argument('-kmcq', '--kmeans-qty', type=int, default=5, help='K-Means Clustering Quantity of Centroids')
    parser.add_argument('-af', '--affiliation-feature', type=str, required=required, help='Affiliation Feature')

    parser.add_argument('-kfcq', '--kfold-qty', type=int, default=10, help='K-Fold Cross Validation Folds Quantity')

    parser.add_argument('-hpo', '--hyper-optim', action='store_true', default=True, help='Hyperparameter k Optimisation Toggle')
    parser.add_argument('-hpoq', '--hyper-optim-range', type=int, default=20, help='Hyperparameter k Optimisation Range')

    parser.add_argument('-plts', '--suppress-all-plots', action='store_true', default=False, help='Suppress All Plots')

    parser.add_argument('-test', '--unittest', action='store_true', default=True, help='Run Unit Tests')

    # parser.print_help()
    args = parser.parse_args()
    args_dict = vars(args)
    print("CLI Arguments: ", args_dict)

    # process_cli = ProcessCLIArgs(args.logistic_toggle, args.linear_toggle, args.knn_toggle)
    process_cli = ProcessCLI(**args_dict)

    event_default = EVENT
    event_mod = process_cli.map_cli_args_to_event_config(event_default)

    output = process_cli.prediction_handler(event_mod)
    pprint(output)