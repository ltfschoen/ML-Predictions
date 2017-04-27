import pandas as pd
from pathlib import Path
import requests

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