import pandas as pd
from pathlib import Path
import requests
from prediction_utils import PredictionUtils
from prediction_config import PredictionConfig

class PredictionData:
    """ Load and Partition DataFrame into Training/Testing for Validation Process """
    def __init__(self):
        self.df_listings = self.load_dataset(None) # Load data set DataFrame (i.e. `None` for all 3723 rows)
        self.training_part = None # Training part of df_listings
        self.testing_part = None # Testing part of df_listings
        self.remove_columns_incorrect_format() # Remove columns with inadequate data (i.e. missing values, non-numeric, non-ordinal, unspecific)
        self.delete_columns_high_incomplete()
        self.retain_columns_low_incomplete_but_strip()
        self.show_columns_incomplete() # Identify quantity of null values per column
        self.normalise_listings()
        self.cleanse_columns()
        self.partition_listings()

    def load_dataset(self, num_rows):
        """ Load downloaded copy of dataset (.csv format) into Pandas Dataframe (DF)
        otherwise load directly from remote endpoint (slower)
        """
        try:
            dataset_file = Path(PredictionConfig.DATASET_LOCAL)
            if dataset_file.is_file():
                return pd.read_csv(PredictionConfig.DATASET_LOCAL, nrows=num_rows)
            else:
                def exists(path):
                    r = requests.head(path)
                    return r.status_code == requests.codes.ok
                if exists(PredictionConfig.DATASET_REMOTE):
                    return pd.read_csv(PredictionConfig.DATASET_REMOTE, nrows=num_rows)
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

    def show_columns_incomplete(self):
        """
        Show quantity of non-null values for each column for inspection.
        Determine columns to remove from the DataFrame (i.e. few non-null)
        """
        df_size = len(self.df_listings)

        # Randomise (not Sorted)
        _temp_df_listings_randomised = PredictionUtils.randomise_dataframe_rows(self.df_listings)

        # Cleanse (whole Set prior to split into Training and Testing parts)
        _temp_df_listings_cleaned = PredictionUtils.clean_price(_temp_df_listings_randomised)
        # print("Length of DataFrame: %r" % (df_size))
        # print("Prediction Data quantity of non-null data per column: %r" % (_temp_df_listings_cleaned.head(n=df_size).info(verbose=True, null_counts=True)))

        df_listings_with_any_null_values = _temp_df_listings_cleaned[_temp_df_listings_cleaned.columns[_temp_df_listings_cleaned.isnull().any()].tolist()]

        print("Prediction Data proportion of null data per column for only columns with any null or NaN values: %r" % (PredictionUtils.get_percentage_missing(df_listings_with_any_null_values)))

    def delete_columns_high_incomplete(self):
        """ Delete Columns where percentage of null or NaN values exceeds MAX_MAJOR_INCOMPLETE value
        These columns may be useless since too many observations missing
        """

        # Iterate over columns in DataFrame
        for name, values in self.df_listings.iteritems():
            # print("%r: %r" % (name, values) )
            if name != "id":
                col_percentage_missing = PredictionUtils.get_percentage_missing(self.df_listings[name])
                if col_percentage_missing > PredictionConfig.MAX_MAJOR_INCOMPLETE:
                    print("Deleting Column %r, as contains too many null values: %r" % (name, col_percentage_missing) )
                    self.df_listings.drop(name, axis=1, inplace=True)

    def retain_columns_low_incomplete_but_strip(self):
        """ Retain Columns where percentage of null or NaN values comprise LESS THAN 1% (0.01) of its rows
        However remove null and NaN rows along requested axis (i.e. axis=0 asks Pandas to drop across DataFrame rows)
        BEWARE: Do not do this for columns where null/NaN values comprise MORE THAN 1% since
        each rows that is removed from a Column represents an Observation, which is shared across the same
        row of all other Columns in the dataset, so that same row is removed across ALL the Columns
        """

        # Iterate over columns in DataFrame
        for name, values in self.df_listings.iteritems():
            col_percentage_missing = PredictionUtils.get_percentage_missing(self.df_listings[name])
            if col_percentage_missing < PredictionConfig.MAX_MINOR_INCOMPLETE:
                # print("Before null/NaN values?: %r" %(self.df_listings[name].isnull().any(axis=0)))
                print("Retained Column: %r, but removed null and NaN valued rows comprising approx. percentage: %r" % (name, col_percentage_missing) )
                self.df_listings.dropna(axis=0, how="any", subset=[name], inplace=True)
                # print("After null/NaN values?: %r" %(self.df_listings[name].isnull().any(axis=0)))

    def normalise_listings(self):
        """ Normalise column values where the column types are normalisable, being of either type int, float64, or floating

        Apply mass Column transformation to Normalise all the feature columns
        in the df_listings DataFrame and assign to a new Dataframe containing just the
        normalised feature columns to normalized_listings.

        Avoid normalizing the "price" TARGET_COLUMN
        """

        # Select only Columns containing type int, float64, floating. Exclude Columns with types Object (O) that includes strings
        df_listings_with_float_or_int_values = self.df_listings.select_dtypes(include=['int', 'float64', 'floating'], exclude=['O'])

        normalized_listings = PredictionUtils.normalise_dataframe(df_listings_with_float_or_int_values)
        normalized_listings[PredictionConfig.TARGET_COLUMN] = self.df_listings[PredictionConfig.TARGET_COLUMN]

        print("Normalised listings completed: %r" % (normalized_listings.head(3)) )

        self.df_listings = normalized_listings

    def cleanse_columns(self):
        """ Cleanse 'price' column """
        self.df_listings = PredictionUtils.clean_price(self.df_listings)

    def partition_listings(self):
        """ Split DataFrame into 2x partitions for the Train/Test Validation Process """
        training_part_end = self.get_training_partitions(self.df_listings)
        try:
            self.training_part = self.df_listings.iloc[0:training_part_end]
            self.testing_part = self.df_listings.iloc[training_part_end:]
        except Exception as e:
            print(e.errno)

    def get_training_partitions(self, df):
        """ Split full dataset into Training and Testing sets (DataFrame partition size proportions) """
        testing_proportion = PredictionConfig.TESTING_PROPORTION
        training_len = int(len(df) - len(df) * testing_proportion) # 75%
        # Cater for test_proportion of 0 to prevent out of bounds exception when later increment
        if training_len >= len(df):
            training_len -= len(df) - 1
        return training_len