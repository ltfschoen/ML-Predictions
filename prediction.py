import matplotlib.pyplot as plt
from prediction_data import PredictionData
from prediction_utils import PredictionUtils

class Prediction:
    """ Machine Learning Model - function that outputs prediction based on input to the model """
    def __init__(self, prediction_data):
        self.prediction_data = prediction_data

    def get_price_prediction(self, model_feature_name):
        _temp_testing_part = self.prediction_data.testing_part
        column_name_predicted_price_feature = "predicted_price_" + model_feature_name
        self.prediction_data.testing_part[column_name_predicted_price_feature] = _temp_testing_part[model_feature_name].apply(lambda x: self.process_price_prediction(model_feature_name, x))
        print("Predicted Prices using Model %r: %r" % (model_feature_name, self.prediction_data.testing_part[column_name_predicted_price_feature]) )

    def process_price_prediction(self, model_feature_name, model_feature_value):
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
        print(_temp_training_part[model_feature_name])
        _temp_training_part[column_name_distance_feature] = PredictionUtils.compare_observations(model_feature_value, _temp_training_part[model_feature_name])

        # Inspect
        # print(_temp_training_part[column_name_distance_feature].value_counts()) # .index.tolist()
        # print(_temp_training_part[_temp_training_part[column_name_distance_feature] == 0][model_feature_name])

        # Randomise and Sort
        _temp_training_part_randomised = PredictionUtils.randomise_dataframe_rows(_temp_training_part)
        _temp_training_part_sorted = PredictionUtils.sort_dataframe_by_feature(_temp_training_part_randomised, column_name_distance_feature)
        # print(_temp_training_part_sorted.iloc[0:10]["price"])

        # Cleanse (Training Set)
        _temp_training_part_cleaned = PredictionUtils.clean_price(_temp_training_part_sorted)
        # print(_temp_training_part_cleaned)

        # Filter
        predicted_price = PredictionUtils.get_nearest_neighbors(_temp_training_part_cleaned, model_feature_name)

        return predicted_price

    def get_mean_absolute_error(self, model_feature_name):
        """ Mean Absolute Error (MAE) calculation """
        _temp_testing_part = self.prediction_data.testing_part

        # Cleanse (Test Set)
        _temp_testing_part_cleaned = PredictionUtils.clean_price(_temp_testing_part)
        # print(_temp_testing_part_cleaned['predicted_price'])
        mae = PredictionUtils.calc_mean_absolute_error(_temp_testing_part_cleaned, model_feature_name)
        print("MAE for Model feature %r: %r: " % (model_feature_name, mae) )
        return mae

    def get_mean_squared_error(self, model_feature_name):
        """ Mean Squared Error (MSE) calculation

        MSE improved prediction accuracy over MAE since penalises predicted values that are further
        from the actual value more that those that are closer to the actual value
        """
        _temp_testing_part = self.prediction_data.testing_part

        # Cleanse (Test Set)
        _temp_testing_part_cleaned = PredictionUtils.clean_price(_temp_testing_part)
        # print(_temp_testing_part_cleaned['predicted_price'])
        mse = PredictionUtils.calc_mean_squared_error(_temp_testing_part_cleaned, model_feature_name)
        print("MSE for Model feature %r: %r: " % (model_feature_name, mse) )
        return mse

    def get_root_mean_squared_error(self, model_feature_name):
        """ Root Mean Squared Error (RMSE) calculation

        RMSE helps understand performance of prediction accuracy over MSE and MAE since
        it takes the square root of MSE so the units matching base unit of the target feature
        """
        _temp_testing_part = self.prediction_data.testing_part

        # Cleanse (Test Set)
        _temp_testing_part_cleaned = PredictionUtils.clean_price(_temp_testing_part)
        # print(_temp_testing_part_cleaned['predicted_price'])
        rmse = PredictionUtils.calc_root_mean_squared_error(_temp_testing_part_cleaned, model_feature_name)
        print("RMSE for Model feature %r: %r: " % (model_feature_name, rmse) )
        return rmse

    def plot(self, model_feature_name):
        """ Plot """
        _temp_testing_part_cleaned = PredictionUtils.clean_price(self.prediction_data.testing_part)
        _temp_testing_part_cleaned.pivot_table(index=model_feature_name, values='price').plot()
        plt.show()

def run():
    prediction_data = PredictionData()
    prediction = Prediction(prediction_data)
    models = ["accommodates", "bathrooms"]
    for index, model_feature_name in enumerate(models):
        prediction.get_price_prediction(model_feature_name)
        mae = prediction.get_mean_absolute_error(model_feature_name)      # MAE
        mse = prediction.get_mean_squared_error(model_feature_name)       # MSE
        rmse = prediction.get_root_mean_squared_error(model_feature_name) # RMSE
        mae_rmse_ratio_prefix = mae / rmse
        print("MAE to RMSE Ratio: %.2f:1" % (mae_rmse_ratio_prefix) )
        prediction.plot(model_feature_name)

run()
