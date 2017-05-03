import pandas as pd
from pathlib import Path
import requests
from prediction_utils import PredictionUtils

class PredictionData:
    """ Load and Partition DataFrame into Training/Testing for Validation Process """
    def __init__(self):
        self.MAX_INCOMPLETE = 0.2 # Percentage
        self.DATASET_LOCAL = "data/listings.csv"
        self.DATASET_REMOTE = "http://data.insideairbnb.com/united-states/dc/washington-dc/2015-10-03/data/listings.csv"
        self.df_listings = self.load_dataset(None) # Load data set DataFrame (i.e. `None` for all 3723 rows)
        self.training_part = None # Training part of df_listings
        self.testing_part = None # Testing part of df_listings
        self.remove_columns_incorrect_format() # Remove columns with inadequate data (i.e. missing values, non-numeric, non-ordinal, unspecific)
        self.delete_columns_high_incomplete()
        self.retain_columns_low_incomplete_but_strip()
        self.show_columns_incomplete() # Identify quantity of null values per column

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

    def remove_columns_incorrect_format(self):
        """
        Return new object with labels in requested axis removed (i.e. axis=1 asks Pandas to drop across DataFrame columns)
        """
        remove_non_numeric_columns = ["room_type", "city", "state"]
        remove_non_ordinal_columns = ["latitude", "longitude", "zipcode"]
        remove_out_of_scope_columns = ["host_response_rate", "host_acceptance_rate", "host_listings_count"]
        remove_columns = remove_non_numeric_columns + \
                         remove_non_ordinal_columns + \
                         remove_out_of_scope_columns
        self.df_listings.drop(remove_columns, axis=1, inplace=True)

    def get_percentage_missing(self, series):
        """ Calculates percentage of NaN values in DataFrame
        :param series: Pandas DataFrame object
        :return: float
        """
        num = series.isnull().sum()
        den = len(series)
        return round(num/den, 2)

    def show_columns_incomplete(self):
        """
        Show quantity of non-null values for each column for inspection.
        Determine columns to remove from the DataFrame (i.e. few non-null)
        """
        _temp_df_listings = self.df_listings
        df_size = len(_temp_df_listings)

        # Randomise (not Sorted)
        _temp_df_listings_randomised = PredictionUtils.randomise_dataframe_rows(_temp_df_listings)

        # Cleanse (whole Set prior to split into Training and Testing parts)
        _temp_df_listings_cleaned = PredictionUtils.clean_price(_temp_df_listings_randomised)
        print("Length of DataFrame: %r" % (df_size))
        # print("Prediction Data quantity of non-null data per column: %r" % (_temp_df_listings_cleaned.head(n=df_size).info(verbose=True, null_counts=True)))

        df_listings_with_any_null_values = _temp_df_listings_cleaned[_temp_df_listings_cleaned.columns[_temp_df_listings_cleaned.isnull().any()].tolist()]

        print("Prediction Data proportion of null data per column for only columns with any null or NaN values: %r" % (self.get_percentage_missing(df_listings_with_any_null_values)))

    def delete_columns_high_incomplete(self):
        """ Delete Columns where percentage of null or NaN values exceeds MAX_INCOMPLETE value
        These columns may be useless since too many observations missing
        """
        _temp_df_listings = self.df_listings

        # Iterate over columns in DataFrame
        for name, values in _temp_df_listings.iteritems():
            # print("%r: %r" % (name, values) )
            if name != "id":
                if self.get_percentage_missing(_temp_df_listings[name]) > self.MAX_INCOMPLETE:
                    print("Deleting Column containing too many null values: %r" % (name) )
                    _temp_df_listings.drop(name, axis=1, inplace=True)

    def retain_columns_low_incomplete_but_strip(self):
        """ Retain Columns where percentage of null or NaN values comprise LESS THAN 1% (0.01) of its rows
        However remove null and NaN rows along requested axis (i.e. axis=0 asks Pandas to drop across DataFrame rows)
        BEWARE: Do not do this for columns where null/NaN values comprise MORE THAN 1% since
        each rows that is removed from a Column represents an Observation, which is shared across the same
        row of all other Columns in the dataset, so that same row is removed across ALL the Columns
        """
        _temp_df_listings = self.df_listings

        # Iterate over columns in DataFrame
        for name, values in _temp_df_listings.iteritems():
            if self.get_percentage_missing(_temp_df_listings[name]) < 0.01:
                print("Retained Column: %r, but removed its null and NaN valued rows" % (name) )
                _temp_df_listings.dropna(axis=0, how="any", subset=[name], inplace=True)

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