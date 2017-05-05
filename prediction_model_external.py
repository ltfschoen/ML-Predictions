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
        self.training_columns = prediction_data.get_training_columns()
        self.target_column = PredictionConfig.TARGET_COLUMN
        self.knn = None

    def generate_knn_model(self, qty_neighbors, algorithm, distance_type):
        self.knn = KNeighborsRegressor(n_neighbors=qty_neighbors, algorithm=algorithm, p=distance_type)

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

        self.generate_knn_model(PredictionConfig.HYPERPARAMETER_FIXED, 'brute', 2)

        _temp_training_part = self.prediction_data.training_part
        X = _temp_training_part[self.training_columns]
        y = _temp_training_part[self.target_column]

        self.knn.fit(X, y)

        _temp_testing_part = self.prediction_data.testing_part
        predictions = self.knn.predict(_temp_testing_part[self.training_columns])

        print("Predictions using Scikit-Learn: %r" % (predictions) )

        mae = median_absolute_error(_temp_testing_part[self.target_column], predictions)
        mse = mean_squared_error(_temp_testing_part[self.target_column], predictions, multioutput='raw_values')
        rmse = math.sqrt(mse)

        print("MAE: %r" % (mae) )
        print("MSE: %r" % (mse[0]) )
        print("RMSE: %r" % (rmse) )

        mae_rmse_ratio_prefix = mae / rmse
        print("MAE to RMSE Ratio: %.2f:1" % (mae_rmse_ratio_prefix) )
        for index, training_model_feature_name in enumerate(self.training_columns):
            PredictionUtils.plot(training_model_feature_name, _temp_testing_part)


    def process_hyperparameter_optimisation(self):
        hyper_param_range = PredictionConfig.HYPERPARAMETER_RANGE
        print("Error: Hyperparameter Optimisation not yet implemented")

def run():
    """
    Scikit-Learn Workflow depending on config chosen
    """
    prediction_data = PredictionData()
    prediction_model_external = PredictionModelExternal(prediction_data)

    if PredictionConfig.HYPERPARAMETER_OPTIMISATION == True:
        prediction_model_external.process_hyperparameter_optimisation()
    else:
        prediction_model_external.process_hyperparameter_fixed()
