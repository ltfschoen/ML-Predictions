#!/usr/bin/env python

import logging
from logging_config import LoggingConfig
from prediction_data import PredictionData
from prediction_config import PredictionConfig
from prediction_utils import PredictionUtils
import prediction_model_manual
import prediction_model_external

def main():
    logging_config = LoggingConfig()
    logging.info('Starting Prediction')
    prediction_config = PredictionConfig()
    prediction_utils = PredictionUtils(prediction_config)
    prediction_data = PredictionData(prediction_config, prediction_utils)

    try:
        if prediction_config.ML_MODEL_KNN == "external":
            prediction_model_external.run(prediction_config, prediction_data, prediction_utils)
        elif prediction_config.ML_MODEL_KNN == "manual":
            prediction_model_manual.run(prediction_config, prediction_data, prediction_utils)
        else:
            raise RuntimeError
    except RuntimeError as e:
        logging.info('Error: No valid KNN Model selected')

    logging.info('Finished Prediction')

if __name__ == '__main__':
    main()