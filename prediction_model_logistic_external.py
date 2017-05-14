from sklearn.linear_model import LogisticRegression
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
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
        self.rmse = None
        self.sensitivity = None
        self.specificity = None

    def plot_logistic_relationships(self, predictions):
        self.prediction_utils.plot_logistic_relationship_comparison(self.prediction_data.df_listings, self.training_columns, predictions)

    def plot_roc(self, fpr, tpr, auc_score):
        self.prediction_utils.plot_receiver_operator_characteristic(fpr, tpr, auc_score)

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
            if self.prediction_config.PLOT_LOGISTIC_RELATIONSHIP_PREDICTION_VS_ACTUAL_FOR_TRAIN_FEATURES_VS_TARGET == True:
                self.plot_logistic_relationships(positive_prediction_proportion)

            # Area Under Curve (AUC) Score
            self.auc_score = roc_auc_score(output, positive_prediction_proportion)
            print("AUC Score: %.2f" % (self.auc_score) )

            # Receiver Operator Characteristic (ROC) Curve
            fpr, tpr, thresholds = roc_curve(output, positive_prediction_proportion)
            if self.prediction_config.PLOT_LOGISTIC_ROC == True:
                self.plot_roc(fpr, tpr, self.auc_score)

            print("Check positive predictions probability accuracy against 'known' Model Training Data:\n %r" % (df[[self.target_column, "predictions_probabilities_positive"]]))

            print("Positive Predictions Probabilities using Scikit-Learn Logistic Regression: %r" % (positive_prediction_proportion) )

            predictions_for_target = self.lr.predict(inputs)
            df["predicted_target_values"] = predictions_for_target
            print("Predicted Output for each Observation: %r" % (df["predicted_target_values"].value_counts()))
            print(df.head(10))

            # Matches between Actual Target row values and Predicted Target row values
            matches = df["predicted_target_values"] == output
            correct_predictions = df[matches]

            # Accuracy of model predictions for given Discrimination Threshold
            self.accuracy = len(correct_predictions) / len(df)

            print("Accuracy of Predictions using Logistic Regression: %.2f" % (self.accuracy) )

            classification = self.prediction_utils.calc_binary_classification(predictions_for_target, output)
            self.sensitivity = classification["sensitivity"]
            self.specificity = classification["specificity"]

            print("Sensitivity of Predictions using Logistic Regression and Binary Classification: %.2f" % (self.sensitivity) )
            print("Specificity of Predictions using Logistic Regression and Binary Classification: %.2f" % (self.specificity) )

            mae = median_absolute_error(df[self.target_column], positive_prediction_proportion)
            mse = mean_squared_error(df[self.target_column], positive_prediction_proportion, multioutput='raw_values')
            self.rmse = math.sqrt(mse)

            print("MAE: %r" % (mae) )
            print("MSE: %r" % (mse[0]) )
            print("RMSE: %r" % (self.rmse) )

            if mae and self.rmse:
                mae_rmse_ratio_prefix = mae / self.rmse
                print("MAE to RMSE Ratio using Logistic Regression: %.2f:1" % (mae_rmse_ratio_prefix) )

            if self.prediction_config.PLOT_INDIVIDUAL_TRAIN_FEATURES_VS_TARGET == True:
                for index, training_model_feature_name in enumerate(self.training_columns):
                    self.prediction_utils.plot(training_model_feature_name, df)
        else:
            print("No Training Columns to use for Logistic Regression. Perhaps they were all bad and removed.")
            rmse = None

        return {
            "rmse": self.rmse,
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "auc_score": self.auc_score
        }

def run(prediction_config, prediction_data, prediction_utils):
    """
    Scikit-Learn Workflow depending on config chosen
    """
    prediction_model_logistic_external = PredictionModelLogisticExternal(prediction_config, prediction_data, prediction_utils)
    return prediction_model_logistic_external.process_logistic_regression()
