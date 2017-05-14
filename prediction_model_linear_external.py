from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
import math

class PredictionModelLinearExternal:
    """ External (Scikit-Learn) Machine Learning Model for Linear Regression - function that outputs prediction based on input to the model """
    def __init__(self, prediction_config, prediction_data, prediction_utils):
        self.prediction_config = prediction_config
        self.prediction_data = prediction_data
        self.prediction_utils = prediction_utils
        self.training_columns = self.prediction_data.training_columns
        self.target_column = self.prediction_config.DATASET_LOCATION[self.prediction_config.DATASET_CHOICE]["target_column"]
        self.model_type = "linear"
        self.response = {}

    def plot_linear_relationships(self, predictions):
        self.prediction_utils.plot_linear_relationship_comparison(self.prediction_data.df_listings, self.training_columns, predictions)

    def process_linear_regression(self):
        """ Linear Regression

        Fit a Machine Learning Model to the data
          - where `input` is matrix with:
            - rows - `n_samples`
            - columns - `n_features`
          - where `output` is:
            - array of `n_samples` when predicting one output
            - matrix of `n_samples` rows and `n_outputs` columns when predicting multiple outputs simultaneously

          - Important Note:
            - Given say a dataset with 400 rows and 10 columns, must pass in matrix of 400 rows and 1 column to predict 1 column
            - Prior to passing `input` to the Fit function, convert the Series/Dataframe objects to a Numpy matrix
              first so Scikit-Learn can convert the input to a Numpy Object
              - WRONG Obtain Numpy array (400 elements) returned from Series using `values` attribute `df["mpg"].values.shape`
              - CORRECT Obtain Numpy matrix object (400 rows, 1 col) returned from Series using `values` attribute `df[["mpg"]].values.shape`
        """
        print("Linear Regression in progress...")

        model = self.prediction_utils.generate_model(self.model_type, None, None, None)
        df = self.prediction_data.df_listings
        inputs = df[self.training_columns]
        if not len(inputs):
            print("No Training Columns to use for Logistic Regression. Perhaps they were all bad and removed.")
            return None

        # Check inputs is Numpy matrix not Numpy array
        print("Shape of inputs to Scikit-Learn Fit function: ", inputs.values.shape)
        output = df[self.target_column]
        model.fit(inputs, output)
        predictions = model.predict(inputs)
        df["predictions"] = predictions
        if self.prediction_config.PLOT_LINEAR_RELATIONSHIP_PREDICTION_VS_ACTUAL_FOR_TRAIN_FEATURES_VS_TARGET == True:
            self.plot_linear_relationships(predictions)
        print("Check predictions accuracy against 'known' Model Training Data:\n %r" % (df[[self.target_column, "predictions"]]))

        print("Predictions using Scikit-Learn Linear Regression: %r" % (predictions) )

        mae = median_absolute_error(df[self.target_column], predictions)
        mse = mean_squared_error(df[self.target_column], predictions, multioutput='raw_values')
        rmse = math.sqrt(mse)

        print("MAE: %r" % (mae) )
        print("MSE: %r" % (mse[0]) )
        print("RMSE: %r" % (rmse) )

        if mae and rmse:
            mae_rmse_ratio_prefix = mae / rmse
            print("MAE to RMSE Ratio using Linear Regression: %.2f:1" % (mae_rmse_ratio_prefix) )

        if self.prediction_config.PLOT_INDIVIDUAL_TRAIN_FEATURES_VS_TARGET == True:
            for index, training_model_feature_name in enumerate(self.training_columns):
                self.prediction_utils.plot(training_model_feature_name, df)

        self.response["pre-hyperparameter_optimisation"] = {
            "model_type": self.model_type,
            "rmse": rmse
        }
        print("Linear Regression Pre-Hyperparameter k Optimisation results: %r" % (self.response))

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
        print("Linear Regression Post-Hyperparameter k Optimisation results: %r" % (self.response))

def run(prediction_config, prediction_data, prediction_utils):
    """
    Scikit-Learn Workflow depending on config chosen
    """
    prediction_model_linear_external = PredictionModelLinearExternal(prediction_config, prediction_data, prediction_utils)
    prediction_model_linear_external.process_linear_regression()

    # Combining K-Fold Cross Validation with Hyperparameter 'k' Optimisation
    if prediction_config.K_FOLD_CROSS_VALIDATION == True:
        prediction_model_linear_external.process_hyperparameter_optimisation()

    return prediction_model_linear_external.response