from prediction_data import PredictionData
from prediction_utils import PredictionUtils
from prediction_config import PredictionConfig

class PredictionModelManual:
    """ Manual Machine Learning Model - function that outputs prediction based on input to the model """
    def __init__(self, prediction_data):
        self.prediction_data = prediction_data

    def get_target_column_prediction(self, model_feature_name):
        _temp_testing_part = self.prediction_data.testing_part
        column_name_predicted_target = "predicted_" + PredictionConfig.TARGET_COLUMN + "_" + model_feature_name
        self.prediction_data.testing_part[column_name_predicted_target] = _temp_testing_part[model_feature_name].apply(lambda x: self.process_target_prediction(model_feature_name, x))
        print("Predicted Target Column (i.e. 'price') using Model %r: %r" % (model_feature_name, self.prediction_data.testing_part[column_name_predicted_target]) )

    def process_target_prediction(self, model_feature_name, model_feature_value):
        """ Compare, Inspect, Randomise, Cleanse, and Filter

        Prior to Randomising and then Sorting, we Inspect and check the value count for "distance" value of 0. Its value is amount of
        other rental listings that also accommodate 3 people, using feature (i.e. "accommodates" or "bathrooms"). Avoid bias (toward just the sort order by "distance"
        column of the data set) when choosing the "nearest neighbors" (all may have distance 0 since only want 5 and there
        are may be around 461 indexes with that distance). Show all listing indexes that have a distance of 0 from my data set
        During Comparison, assign distance values to new "distance" column of Data Frame Series object.
        During Inspection, use the Panda Series method value_counts to display unique value counts for each "distance" column
        """

        column_name_distance_feature = "distance_" + model_feature_name

        # Compare
        _temp_training_part = self.prediction_data.training_part
        # print(_temp_training_part[model_feature_name])
        _temp_training_part[column_name_distance_feature] = PredictionUtils.compare_observations(model_feature_value, _temp_training_part[model_feature_name])

        # Inspect
        # print(_temp_training_part[column_name_distance_feature].value_counts()) # .index.tolist()
        # print(_temp_training_part[_temp_training_part[column_name_distance_feature] == 0][model_feature_name])

        # Randomise and Sort
        _temp_training_part_randomised = PredictionUtils.randomise_dataframe_rows(_temp_training_part)
        _temp_training_part_sorted = PredictionUtils.sort_dataframe_by_feature(_temp_training_part_randomised, column_name_distance_feature)
        # print(_temp_training_part_sorted.iloc[0:10][PredictionConfig.TARGET_COLUMN])

        # Filter
        predicted_target = PredictionUtils.get_nearest_neighbors(_temp_training_part_sorted, model_feature_name)

        return predicted_target

    def get_mean_absolute_error(self, model_feature_name):
        """ Mean Absolute Error (MAE) calculation """
        mae = PredictionUtils.calc_mean_absolute_error(self.prediction_data.testing_part, model_feature_name)
        print("MAE for Model feature %r: %r: " % (model_feature_name, mae) )
        return mae

    def get_mean_squared_error(self, model_feature_name):
        """ Mean Squared Error (MSE) calculation

        MSE improved prediction accuracy over MAE since penalises predicted values that are further
        from the actual value more that those that are closer to the actual value
        """
        mse = PredictionUtils.calc_mean_squared_error(self.prediction_data.testing_part, model_feature_name)
        print("MSE for Model feature %r: %r: " % (model_feature_name, mse) )
        return mse

    def get_root_mean_squared_error(self, model_feature_name):
        """ Root Mean Squared Error (RMSE) calculation

        RMSE helps understand performance of prediction accuracy over MSE and MAE since
        it takes the square root of MSE so the units matching base unit of the target feature
        """
        rmse = PredictionUtils.calc_root_mean_squared_error(self.prediction_data.testing_part, model_feature_name)
        print("RMSE for Model feature %r: %r: " % (model_feature_name, rmse) )
        return rmse

def run():
    prediction_data = PredictionData()
    prediction_model_manual = PredictionModelManual(prediction_data)
    for index, training_model_feature_name in enumerate(prediction_data.get_training_columns()):
        prediction_model_manual.get_target_column_prediction(training_model_feature_name)
        mae = prediction_model_manual.get_mean_absolute_error(training_model_feature_name)      # MAE
        mse = prediction_model_manual.get_mean_squared_error(training_model_feature_name)       # MSE
        rmse = prediction_model_manual.get_root_mean_squared_error(training_model_feature_name) # RMSE
        mae_rmse_ratio_prefix = mae / rmse
        print("MAE to RMSE Ratio: %.2f:1" % (mae_rmse_ratio_prefix) )
        PredictionUtils.plot(training_model_feature_name, prediction_data.testing_part)
