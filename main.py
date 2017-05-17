#!/usr/bin/env python

from __future__ import print_function # Python 2 or 3
from pprint import pprint
import logging
from logging_config import LoggingConfig
from prediction_data import PredictionData
from prediction_config import PredictionConfig
from prediction_utils import PredictionUtils
import prediction_model_knn_manual
import prediction_model_knn_external
import prediction_model_linear_external
import prediction_model_logistic_external
from input_event import EVENT

def main(event, context):
    logging_config = LoggingConfig()
    logging.info('Starting Prediction')
    prediction_config = PredictionConfig(event, context)
    prediction_utils = PredictionUtils(prediction_config)
    prediction_data = PredictionData(prediction_config, prediction_utils)

    try:
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
            return prediction_model_knn_external.run(prediction_config, prediction_data, prediction_utils)
        elif prediction_config.ML_MODEL_KNN == "manual":
            return prediction_model_knn_manual.run(prediction_config, prediction_data, prediction_utils)
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

if __name__ == '__main__':
    output = prediction_handler(EVENT)
    pprint(output)