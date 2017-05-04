from prediction_data import PredictionData
from prediction_utils import PredictionUtils
from prediction_config import PredictionConfig
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
import math

class PredictionModelExternal:
    """ External (Scikit-Learn) Machine Learning Model - function that outputs prediction based on input to the model """
    def __init__(self, prediction_data):
        self.prediction_data = prediction_data
        self.knn = None

    def generate_knn_model(self, qty_neighbors, algorithm, distance_type):
        self.knn = KNeighborsRegressor(n_neighbors=qty_neighbors, algorithm=algorithm, p=distance_type)

def run():
    prediction_data = PredictionData()
    prediction_model_external = PredictionModelExternal(prediction_data)

    """
    Scikit-Learn Workflow
    """

    # Step 1: Create instance of K-Nearest-Neighbors Machine Learning Model class where p=2 is Euclidean Distance
    prediction_model_external.generate_knn_model(5, 'brute', 2)

    # Step 2: Fit the Model using by specifying data for K-Nearest-Neighbor Model to use:
    #   - X as Training data (i.e. DataFrame "feature" Columns from Training data)
    #   - y as Target values (i.e. DataFrame's Target Column)

    # X for `fit` function is matrix-like object, containing
    # columns of interest from Training set (to use to make Predictions).

    print("Multiple Training features include: %r" % (prediction_data.get_training_columns() ) )

    _temp_training_part = prediction_model_external.prediction_data.training_part
    X = _temp_training_part[prediction_data.get_training_columns()]
    print(X)

    # y for `fit` function is list-like object, containing
    # just TARGET_COLUMN, `price`.
    y = _temp_training_part[PredictionConfig.TARGET_COLUMN]
    print(y)

    # X and y are passed into `fit` method of Scikit-Learn.
    # Warning: DO NOT pass in data containing the following else Error occurs:
    #   - Missing values
    #   - Non-numerical values
    prediction_model_external.knn.fit(X, y)

    # Scikit-Learn's `predict` function called to make Predictions on
    # Columns of test_df that returns a NumPy array of Predicted "price" TARGET_COLUMN values
    _temp_testing_part = prediction_model_external.prediction_data.training_part
    predictions = prediction_model_external.knn.predict(_temp_testing_part[prediction_data.get_training_columns()])

    print("Predictions using Scikit-Learn: %r" % (predictions) )

    # Calculate MAE float values for each individual Target, where least loss "best" values are 0
    multiple_features_mae = median_absolute_error(_temp_testing_part[PredictionConfig.TARGET_COLUMN], predictions)

    # Calculate MSE float values for each individual Target, where least loss "best" values are 0
    multiple_features_mse = mean_squared_error(_temp_testing_part[PredictionConfig.TARGET_COLUMN], predictions, multioutput='raw_values')

    # Calculate RMSE
    multiple_features_rmse = math.sqrt(multiple_features_mse)

    print("MAE (Multiple features): %r" % (multiple_features_mae) )
    print("MSE (Multiple features): %r" % (multiple_features_mse[0]) )
    print("RMSE (Multiple features): %r" % (multiple_features_rmse) )

    multiple_features_mae_rmse_ratio_prefix = multiple_features_mae / multiple_features_rmse
    print("MAE to RMSE Ratio (Multiple features): %.2f:1" % (multiple_features_mae_rmse_ratio_prefix) )
    for index, training_model_feature_name in enumerate(prediction_data.get_training_columns()):
        PredictionUtils.plot(training_model_feature_name, _temp_testing_part)
