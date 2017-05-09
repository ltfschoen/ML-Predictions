#!/usr/bin/env python

from __future__ import print_function # Python 2 or 3
from pprint import pprint
import logging
from logging_config import LoggingConfig
from prediction_data import PredictionData
from prediction_config import PredictionConfig
from prediction_utils import PredictionUtils
import prediction_model_manual
import prediction_model_external

def main(event, context):
    logging_config = LoggingConfig()
    logging.info('Starting Prediction')
    prediction_config = PredictionConfig(event, context)
    prediction_utils = PredictionUtils(prediction_config)
    prediction_data = PredictionData(prediction_config, prediction_utils)

    try:
        if prediction_config.ML_MODEL_KNN == "scikit":
            return prediction_model_external.run(prediction_config, prediction_data, prediction_utils)
        elif prediction_config.ML_MODEL_KNN == "manual":
            return prediction_model_manual.run(prediction_config, prediction_data, prediction_utils)
        else:
            raise RuntimeError
    except RuntimeError as e:
        logging.info('Error: No valid KNN Model selected')

    logging.info('Finished Prediction')

def get_log_stream(event, context):
    """ AWS Lambda Context """

    print("Log stream name:", context.log_stream_name)
    print("Log group name:",  context.log_group_name)
    print("Request ID:",context.aws_request_id)
    print("Mem. limits(MB):", context.memory_limit_in_mb)
    print("Time remaining (MS):", context.get_remaining_time_in_millis())

def prediction_handler(event, context=None):
    """ AWS Lambda Handler

    Event argument only provided when using AWS Lambda
    """

    if context:
        get_log_stream(context)

    prediction = None

    if event:
        prediction = main(event, context)
    else:
        print("Error: Missing event object is required argument of handler")

    return {
        'prediction' : prediction
    }

EVENT = {
    "model_workflow_for_knn_algorithm": "scikit",
    "hyperparameter_optimisation_toggle": True,
    "hyperparameter_range": 20,
    "hyperparameter_quantity_fixed": 5,
    "min_training_features": 3,
    "min_percentage_incomplete_observations_to_remove_column": 0.2,
    "max_percentage_incomplete_observations_to_retain_column_and_remove_incomplete_slice": 0.02,
    "k_fold_cross_validation_toggle": True,
    "k_folds_quantity": 10,
    "k_folds_workflow": "scikit",
    "dataset_selected": "car-listings",
    "dataset_config": {
        "rental-property-listings": {
            "local": "data/listings.csv",
            "remote": "http://data.insideairbnb.com/united-states/dc/washington-dc/2015-10-03/data/listings.csv",
            "labels": "",
            "training_columns": ["accommodates", "bedrooms", "bathrooms", "number_of_reviews"],
            "target_column": "price",
            "cleanse_columns_price_format": ["price"],
            "convert_columns_words_to_digits": []
        },
        "car-listings": {
            "local": "data/imports-85.data",
            "remote": "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
            "labels": "symboling,normalized-losses,make,fuel-type,aspiration,num-of-doors,body-style,drive-wheels,engine-location,wheel-base,length,width,height,curb-weight,engine-type,num-of-cylinders,engine-size,fuel-system,bore,stroke,compression-ratio,horsepower,peak-rpm,city-mpg,highway-mpg,price",
            "training_columns": ["num-of-doors", "curb-weight", "horsepower", "city-mpg", "highway-mpg"],
            "target_column": "price",
            "cleanse_columns_price_format": ["price"],
            "convert_columns_words_to_digits": ["num-of-doors", "num-of-cylinders"]
        }
    }
}

if __name__ == '__main__':
    output = prediction_handler(EVENT)
    pprint(output)