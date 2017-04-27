import matplotlib.pyplot as plt
from prediction_data import PredictionData
from prediction_utils import PredictionUtils

class Prediction:
    """ Machine Learning Model - function that outputs prediction based on input to the model """
    def __init__(self, prediction_data):
        self.prediction_data = prediction_data

    def get_price_prediction(self):
        _temp_testing_part = self.prediction_data.testing_part
        self.prediction_data.testing_part["predicted_price"] = _temp_testing_part['accommodates'].apply(lambda x: self.process_price_prediction(x))
        print("Predicted Prices: ", self.prediction_data.testing_part["predicted_price"] )

        # Plot
        _temp_testing_part_cleaned = PredictionUtils.clean_price(self.prediction_data.testing_part)
        _temp_testing_part_cleaned.pivot_table(index='accommodates', values='price').plot()
        plt.show()

    def process_price_prediction(self, accommodates_qty):
        """ Compare, Inspect, Randomise, Cleanse, and Filter

        Prior to Randomising and then Sorting, we Inspect and check the value count for "distance" value of 0. Its value is amount of
        other rental listings that also accommodate 3 people, using feature "accommodates". Avoid bias (toward just the sort order by "distance"
        column of the data set) when choosing the "nearest neighbors" (all may have distance 0 since only want 5 and there
        are may be around 461 indexes with that distance). Show all listing indexes that have a distance of 0 from my data set

        During Comparison, assign distance values to new "distance" column of Data Frame Series object.

        During Inspection, use the Panda Series method value_counts to display unique value counts for each "distance" column
        """

        # Compare
        _temp_training_part = self.prediction_data.training_part
        _temp_training_part["distance"] = PredictionUtils.compare_observations(accommodates_qty, _temp_training_part["accommodates"])

        # Inspect
        # print(_temp_training_part["distance"].value_counts()) # .index.tolist()
        # print(_temp_training_part[_temp_training_part["distance"] == 0]["accommodates"])

        # Randomise
        _temp_training_part_randomised = PredictionUtils.randomise_dataframe_rows(_temp_training_part)
        _temp_training_part_sorted = PredictionUtils.sort_dataframe_by_feature(_temp_training_part_randomised, "distance")
        # print(_temp_training_part_sorted.iloc[0:10]["price"])

        # Cleanse
        _temp_training_part_cleaned = PredictionUtils.clean_price(_temp_training_part_sorted)
        # print(_temp_training_part_cleaned)

        # Filter
        predicted_price = PredictionUtils.get_nearest_neighbors(_temp_training_part_cleaned)

        return predicted_price

def run():
    prediction_data = PredictionData()
    prediction = Prediction(prediction_data)
    prediction.get_price_prediction()

run()
