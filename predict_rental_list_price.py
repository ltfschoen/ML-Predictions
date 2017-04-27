import pandas as pd
import numpy as np
import math
from pathlib import Path
import requests
import matplotlib.pyplot as plt

"""
Given you have a rental listing that accommodates up to 3 rooms.
And given a data set that contains features (column attributes) of other rental listings.
Find the optimum rental listing price using similarity metrics
"""


class PredictionData:
    """ Load and Partition DataFrame into Training/Testing for Validation Process """
    def __init__(self):
        self.DATASET_LOCAL = "data/listings.csv"
        self.DATASET_REMOTE = "http://data.insideairbnb.com/united-states/dc/washington-dc/2015-10-03/data/listings.csv"
        self.df_listings = self.load_dataset(None) # Load data set DataFrame (i.e. `None` for all 3723 rows)
        self.training_part = None # Training part of df_listings
        self.testing_part = None # Testing part of df_listings
        self.partition_listings()

    def load_dataset(self, num_rows):
        """ Load downloaded copy of dataset (.csv format) into Pandas Dataframe (DF)
        otherwise load directly from remote endpoint (slower)
        """
        try:
            dataset_file = Path(self.DATASET_LOCAL)
            if dataset_file.is_file():
                return pd.read_csv(self.DATASET_LOCAL, nrows=num_rows)
            else:
                def exists(path):
                    r = requests.head(path)
                    return r.status_code == requests.codes.ok
                if exists(self.DATASET_REMOTE):
                    return pd.read_csv(self.DATASET_REMOTE, nrows=num_rows)
            return None
        except Exception as e:
            print(e.errno)

    def partition_listings(self):
        """ Split DataFrame into 2x partitions for the Train/Test Validation Process """
        training_part_end = self.get_training_partitions(self.df_listings)
        try:
            self.training_part = self.df_listings.iloc[0:training_part_end]
            self.testing_part = self.df_listings.iloc[training_part_end:]
        except Exception as e:
            print(e.errno)

    def get_training_partitions(self, df):
        """ Split DataFrame partition size proportions:
            - Training Set - 75% of rows
            - Test - Set 25% of rows"""
        testing_proportion = 0.25 # Between 0 and 1
        training_len = int(len(df) - len(df) * testing_proportion) # 75%
        # Cater for test_proportion of 0 to prevent out of bounds exception when later increment
        if training_len >= len(df):
            training_len -= len(df) - 1
        return training_len


class PredictionUtils(object):
    """ Utility functions """

    @staticmethod
    def calc_euclidean_dist(val1, val2):
        """ Euclidean Distance equation to compare values of different data sets """
        return int(math.sqrt(abs(val1 - val2)**2))

    @staticmethod
    def compare_observations(obs1, obs2):
        """ Similarity Metric compares two observations' data set features (columns)
        and returns distance (difference). Compare value of feature "accommodates" in across DataFrame Series
        """
        return obs2.apply(lambda x: PredictionUtils.calc_euclidean_dist(x, obs1))

    @staticmethod
    def randomise_dataframe_rows(df):
        """ Randomise ordering of DataFrame.
        Return a NumPy array of shuffled index values using `np.random.permutation`
        Return a new Dataframe containing the shuffled order using `loc[]`
        `seed(1)` reproduces random same results when share and run same code by others
        """
        np.random.seed(1)
        return df.loc[np.random.permutation(len(df))]

    @staticmethod
    def sort_dataframe_by_feature(df, feature):
        """ Sort DataFrame by feature (default ascending).
        Sort the DataFrame by "distance" column so there will be random order across the
        rows at the top of the list (having same lowest distance).
        """
        return df.sort_values(feature)

    @staticmethod
    def clean_price(df):
        """ Clean "price" column removing `$` and `,` chars. Convert column from text to float. """
        def replace_bad_chars(row):
            row = row.replace(",", "")
            row = row.replace("$", "")
            row = float(row) # .astype('float')
            return row
        df["price"] = df["price"].apply(lambda row: replace_bad_chars(row))
        return df

    @staticmethod
    def get_nearest_neighbors(df):
        """ Filter range of nearest neighbors to select of recommended prices to charge per night for a rental listing based
        on average price of other listings that accommodate that same or similar size (i.e. 3 or so people).
        """
        print("Predicted Price (Avg of Nearest): %.2f (with Avg Accommodates: %r) " % (df.iloc[0:5]["price"].mean(), df.iloc[0:5]["accommodates"].mean()) )
        return df.iloc[0:5]["price"].mean()

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
