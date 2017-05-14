from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
import math
import numpy as np

class PredictionModelKNNExternal:
    """ External (Scikit-Learn) Machine Learning Model for KNN Regression - function that outputs prediction based on input to the model """
    def __init__(self, prediction_config, prediction_data, prediction_utils):
        self.prediction_config = prediction_config
        self.prediction_data = prediction_data
        self.prediction_utils = prediction_utils
        self.training_columns = self.prediction_data.training_columns
        self.target_column = self.prediction_config.DATASET_LOCATION[self.prediction_config.DATASET_CHOICE]["target_column"]
        self.model_type = "knn"

    def process_hyperparameter_fixed(self):
        """
        Step 1: Create instance of K-Nearest-Neighbors Machine Learning Model class where p=2 is Euclidean Distance
        Step 2: Fit the Model using by specifying data for K-Nearest-Neighbor Model to use:
            - X as Training data (i.e. DataFrame "feature" Columns from Training data)
            - y as Target values (i.e. DataFrame's Target Column)

            X argument of `fit` function is matrix-like object, containing cols of interest from Training set (to make predictions)
            y argument of `fit` function is list-like object, containing just TARGET_COLUMN, `price`.

            X and y are passed into `fit` method of Scikit-Learn.
                Warning: DO NOT pass in data containing the following else Error occurs:
                    - Missing values
                    - Non-numerical values

        Step 3: Scikit-Learn's `predict` function called to make predictions on cols of test_df.
            Returns NumPy array of predicted "price" TARGET_COLUMN values

        Step 4: Calculate MAE, MSE, and RMSE float values for each individual Target, where least loss "best" values are 0
        """
        print("Training features include: %r" % (self.training_columns) )

        training_column_names = self.training_columns
        feature_combo = '__'.join(training_column_names)

        model = self.prediction_utils.generate_model(self.model_type, self.prediction_config.HYPERPARAMETER_FIXED, 'brute', 2)

        _temp_training_part = self.prediction_data.training_part
        X = _temp_training_part[self.training_columns]
        y = _temp_training_part[self.target_column]

        model.fit(X, y)

        _temp_testing_part = self.prediction_data.testing_part
        predictions = model.predict(_temp_testing_part[self.training_columns])

        print("Predictions using Scikit-Learn KNN Regression: %r" % (predictions) )

        mae = median_absolute_error(_temp_testing_part[self.target_column], predictions)
        mse = mean_squared_error(_temp_testing_part[self.target_column], predictions, multioutput='raw_values')
        rmse = math.sqrt(mse)

        print("MAE: %r" % (mae) )
        print("MSE: %r" % (mse[0]) )
        print("RMSE: %r" % (rmse) )

        if mae and rmse:
            mae_rmse_ratio_prefix = mae / rmse
            print("MAE to RMSE Ratio: %.2f:1" % (mae_rmse_ratio_prefix) )

        if self.prediction_config.PLOT_INDIVIDUAL_TRAIN_FEATURES_VS_TARGET == True:
            for index, training_model_feature_name in enumerate(self.training_columns):
                self.prediction_utils.plot(training_model_feature_name, _temp_testing_part)

        return {
            "feature_names": feature_combo,
            "rmse": rmse,
            "k_neighbors_qty": self.prediction_config.HYPERPARAMETER_FIXED,
            "k_folds_qty": None,
            "k_fold_cross_validation_toggle": False
        }

    def process_hyperparameter_optimisation(self):
        """ Hyperparameter 'k' Optimisation """
        print("Hyperparameter k Optimisation in progress...")

        feature_combos = self.prediction_utils.generate_combinations_of_features(self.training_columns)

        if self.prediction_config.K_FOLD_CROSS_VALIDATION == False:

            feature_combos_rmse_for_hyperparams = dict()
            _temp_training_part = self.prediction_data.training_part
            _temp_testing_part = self.prediction_data.testing_part

            for idx1, feature_combo in enumerate(feature_combos):
                feature_combo_key = '__'.join(feature_combo)
                feature_combos_rmse_for_hyperparams[feature_combo_key] = list()
                for idx2, qty_neighbors in enumerate(self.prediction_config.HYPERPARAMETER_RANGE):
                    model = self.prediction_utils.generate_model(self.model_type, qty_neighbors, 'brute', 2)
                    X = _temp_training_part[list(feature_combo)]
                    y = _temp_training_part[self.target_column]
                    model.fit(X, y)
                    predictions = model.predict(_temp_testing_part[list(feature_combo)])
                    mse = mean_squared_error(_temp_testing_part[self.target_column], predictions, multioutput='raw_values')
                    rmse = math.sqrt(mse[0])
                    feature_combos_rmse_for_hyperparams[feature_combo_key].append(rmse)

        # Combining K-Fold Cross Validation with Hyperparameter 'k' Optimisation
        else:
            feature_combos_rmse_for_hyperparams = self.prediction_utils.k_fold_cross_validation(self.model_type, self.prediction_data.df_listings, feature_combos)

        optimisation_results = self.prediction_utils.hyperparameter_k_optimisation(feature_combos_rmse_for_hyperparams, self.model_type, None)

        return {
            "feature_names": optimisation_results["feature_combo_name_with_lowest_rmse"],
            "rmse": optimisation_results["lowest_rmse"],
            "k_neighbors_qty": optimisation_results["k_value_of_lowest_rmse"],
            "k_folds_qty": self.prediction_config.K_FOLDS,
            "k_fold_cross_validation_toggle": self.prediction_config.K_FOLD_CROSS_VALIDATION
        }

def run(prediction_config, prediction_data, prediction_utils):
    """
    Scikit-Learn Workflow depending on config chosen
    """
    prediction_model_knn_external = PredictionModelKNNExternal(prediction_config, prediction_data, prediction_utils)

    if prediction_config.HYPERPARAMETER_OPTIMISATION == True:
        return prediction_model_knn_external.process_hyperparameter_optimisation()
    else:
        return prediction_model_knn_external.process_hyperparameter_fixed()
