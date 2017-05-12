from sklearn.linear_model import LogisticRegression
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
import math
import numpy as np

class PredictionModelLogisticExternal:
    """ External (Scikit-Learn) Machine Learning Model for Logistic Regression """
    def __init__(self, prediction_config, prediction_data, prediction_utils):
        self.prediction_config = prediction_config
        self.prediction_data = prediction_data
        self.prediction_utils = prediction_utils
        self.training_columns = self.prediction_data.training_columns
        self.target_column = self.prediction_config.DATASET_LOCATION[self.prediction_config.DATASET_CHOICE]["target_column"]
        self.lr = None

    def plot_logistic_relationships(self, predictions):
        self.prediction_utils.plot_logistic_relationship_comparison(self.prediction_data.df_listings, self.training_columns, predictions)

    def generate_lr_model(self):
        # LogisticRegression class from Scikit-Learn
        self.lr = LogisticRegression(class_weight='balanced')

    def process_logistic_regression(self):
        """ Logistic Regression
        """
        print("Logistic Regression in progress...")

        self.generate_lr_model()
        df = self.prediction_data.df_listings
        inputs = df[self.training_columns]
        if len(inputs):
            # Check inputs is Numpy matrix not Numpy array
            print("Shape of inputs to Scikit-Learn Fit function: ", inputs.values.shape)
            output = df[self.target_column]
            # Overcome the following error that occurs when ALL Target values are either 0 or 1
            # ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0
            if output.between(0,0).all():
                positive_prediction_proportion = [0] * len(df)
            elif output.between(1,1).all():
                positive_prediction_proportion = [1] * len(df)
            else:
                self.lr.fit(inputs, output)
                predictions_probabilities = self.lr.predict_proba(inputs)
                # Note: Sum of Positive and Non-Positive Prediction Proportions add up to 1 for each observation
                #  - Positive Prediction 1 Proportion predictions_probabilities[:,1]
                #  - Non-Prediction 0 Proportion predictions_probabilities[:,0]
                positive_prediction_proportion = predictions_probabilities[:,1]
            df["predictions_probabilities_positive"] = positive_prediction_proportion
            self.plot_logistic_relationships(positive_prediction_proportion)
            print("Check positive predictions probability accuracy against 'known' Model Training Data:\n %r" % (df[[self.target_column, "predictions_probabilities_positive"]]))

            print("Positive Predictions Probabilities using Scikit-Learn Logistic Regression: %r" % (positive_prediction_proportion) )

            predictions_for_labels = self.lr.predict(inputs)
            df["predictioned_target_values"] = predictions_for_labels
            print("Predicted Output for each Observation: %r" % (df["predictioned_target_values"].value_counts()))
            print(df.head(10))

            # Matches between Actual Target row values and Predicted Target row values
            matches = df["predictioned_target_values"] == output
            correct_predictions = df[matches]

            # Accuracy of model predictions for given Discrimination Threshold
            accuracy = len(correct_predictions) / len(df)

            print("Accuracy of Predictions using Logistic Regression: %.2f" % (accuracy) )

            mae = median_absolute_error(df[self.target_column], positive_prediction_proportion)
            mse = mean_squared_error(df[self.target_column], positive_prediction_proportion, multioutput='raw_values')
            rmse = math.sqrt(mse)

            print("MAE: %r" % (mae) )
            print("MSE: %r" % (mse[0]) )
            print("RMSE: %r" % (rmse) )

            if mae and rmse:
                mae_rmse_ratio_prefix = mae / rmse
                print("MAE to RMSE Ratio using Logistic Regression: %.2f:1" % (mae_rmse_ratio_prefix) )

            for index, training_model_feature_name in enumerate(self.training_columns):
                self.prediction_utils.plot(training_model_feature_name, df)
        else:
            print("No Training Columns to use for Logistic Regression. Perhaps they were all bad and removed.")
            rmse = None

        return {
            "rmse": rmse,
            "accuracy": accuracy
        }

def run(prediction_config, prediction_data, prediction_utils):
    """
    Scikit-Learn Workflow depending on config chosen
    """
    prediction_model_logistic_external = PredictionModelLogisticExternal(prediction_config, prediction_data, prediction_utils)
    return prediction_model_logistic_external.process_logistic_regression()
