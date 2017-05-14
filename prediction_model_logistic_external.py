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
        self.model_type = "logistic"
        self.response = {}

    def plot_logistic_relationships(self, predictions):
        self.prediction_utils.plot_logistic_relationship_comparison(self.prediction_data.df_listings, self.training_columns, predictions)

    def plot_roc(self, fpr, tpr, auc_score):
        self.prediction_utils.plot_receiver_operator_characteristic(fpr, tpr, auc_score)

    def process_logistic_regression(self):
        """ Logistic Regression
        """
        print("Logistic Regression in progress...")

        model = self.prediction_utils.generate_model(self.model_type, None, None, None)
        df = self.prediction_data.df_listings
        inputs = df[self.training_columns]

        if not len(inputs):
            print("No Training Columns to use for Logistic Regression. Perhaps they were all bad and removed.")
            return None

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
            model.fit(inputs, output)
            predictions_probabilities = model.predict_proba(inputs)
            # Note: Sum of Positive and Non-Positive Prediction Proportions add up to 1 for each observation
            #  - Positive Prediction 1 Proportion predictions_probabilities[:,1]
            #  - Non-Prediction 0 Proportion predictions_probabilities[:,0]
            positive_prediction_proportion = predictions_probabilities[:,1]
        df["predictions_probabilities_positive"] = positive_prediction_proportion
        if self.prediction_config.PLOT_LOGISTIC_RELATIONSHIP_PREDICTION_VS_ACTUAL_FOR_TRAIN_FEATURES_VS_TARGET == True:
            self.plot_logistic_relationships(positive_prediction_proportion)

        # Area Under Curve (AUC) Score
        auc_score = roc_auc_score(output, positive_prediction_proportion)
        print("AUC Score: %.2f" % (auc_score) )

        # Receiver Operator Characteristic (ROC) Curve
        fpr, tpr, thresholds = roc_curve(output, positive_prediction_proportion)
        if self.prediction_config.PLOT_LOGISTIC_ROC == True:
            self.plot_roc(fpr, tpr, auc_score)

        print("Check positive predictions probability accuracy against 'known' Model Training Data:\n %r" % (df[[self.target_column, "predictions_probabilities_positive"]]))

        print("Positive Predictions Probabilities using Scikit-Learn Logistic Regression: %r" % (positive_prediction_proportion) )

        predictions_for_target = model.predict(inputs)
        df["predicted_target_values"] = predictions_for_target
        print("Predicted Output for each Observation: %r" % (df["predicted_target_values"].value_counts()))
        print(df.head(10))

        # Matches between Actual Target row values and Predicted Target row values
        matches = df["predicted_target_values"] == output
        correct_predictions = df[matches]

        # Accuracy of model predictions for given Discrimination Threshold
        accuracy = len(correct_predictions) / len(df)

        print("Accuracy of Predictions using Logistic Regression: %.2f" % (accuracy) )

        classification = self.prediction_utils.calc_binary_classification(predictions_for_target, output)
        sensitivity = classification["sensitivity"]
        specificity = classification["specificity"]

        print("Sensitivity of Predictions using Logistic Regression and Binary Classification: %.2f" % (sensitivity) )
        print("Specificity of Predictions using Logistic Regression and Binary Classification: %.2f" % (specificity) )

        mae = median_absolute_error(df[self.target_column], positive_prediction_proportion)
        mse = mean_squared_error(df[self.target_column], positive_prediction_proportion, multioutput='raw_values')
        rmse = math.sqrt(mse)

        print("MAE: %r" % (mae) )
        print("MSE: %r" % (mse[0]) )
        print("RMSE: %r" % (rmse) )

        if mae and rmse:
            mae_rmse_ratio_prefix = mae / rmse
            print("MAE to RMSE Ratio using Logistic Regression: %.2f:1" % (mae_rmse_ratio_prefix) )

        if self.prediction_config.PLOT_INDIVIDUAL_TRAIN_FEATURES_VS_TARGET == True:
            for index, training_model_feature_name in enumerate(self.training_columns):
                self.prediction_utils.plot(training_model_feature_name, df)

        self.response["pre-hyperparameter_optimisation"] = {
            "model_type": self.model_type,
            "rmse": rmse,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auc_score": auc_score
        }
        print("Logistic Regression Pre-Hyperparameter k Optimisation results: %r" % (self.response))

    def process_hyperparameter_optimisation(self):
        """ Hyperparameter 'k' Optimisation """
        print("Hyperparameter k Optimisation in progress...")

        feature_combos = self.prediction_utils.generate_combinations_of_features(self.training_columns)
        feature_combos_rmse_for_hyperparams = self.prediction_utils.k_fold_cross_validation(self.model_type, self.prediction_data.df_listings, feature_combos)
        optimisation_results = self.prediction_utils.hyperparameter_k_optimisation(feature_combos_rmse_for_hyperparams, self.model_type, self.response["pre-hyperparameter_optimisation"])
        self.response["post-hyperparameter_optimisation"] = {
            "model_type": self.model_type,
            "feature_names": optimisation_results["feature_combo_name_with_lowest_rmse"],
            "rmse": optimisation_results["lowest_rmse"],
            "k_neighbors_qty": optimisation_results["k_value_of_lowest_rmse"],
            "k_folds_qty": self.prediction_config.K_FOLDS,
            "k_fold_cross_validation_toggle": self.prediction_config.K_FOLD_CROSS_VALIDATION
        }
        print("Logistic Regression Post-Hyperparameter k Optimisation results: %r" % (self.response))

def run(prediction_config, prediction_data, prediction_utils):
    """
    Scikit-Learn Workflow depending on config chosen
    """
    prediction_model_logistic_external = PredictionModelLogisticExternal(prediction_config, prediction_data, prediction_utils)
    prediction_model_logistic_external.process_logistic_regression()

    # Combining K-Fold Cross Validation with Hyperparameter 'k' Optimisation
    if prediction_config.K_FOLD_CROSS_VALIDATION == True:
        prediction_model_logistic_external.process_hyperparameter_optimisation()

    return prediction_model_logistic_external.response
