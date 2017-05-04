#!/usr/bin/env python

import logging
from logging_config import LoggingConfig
from prediction_config import PredictionConfig
import prediction_model_manual
import prediction_model_external

def main():
    logging_config = LoggingConfig()
    logging.info('Starting Prediction')

    try:
        if PredictionConfig.ML_MODEL_KNN == "external":
            prediction_model_external.run()
        elif PredictionConfig.ML_MODEL_KNN == "manual":
            prediction_model_manual.run()
        else:
            raise RuntimeError
    except RuntimeError as e:
        logging.info('Error: No valid KNN Model selected')

    logging.info('Finished Prediction')

if __name__ == '__main__':
    main()