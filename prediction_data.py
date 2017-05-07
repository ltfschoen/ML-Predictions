import pandas as pd
from pathlib import Path
import requests
from lib.build_folds import build_folds

class PredictionData:
    """ Load and Partition DataFrame into Training/Testing for Validation Process """
    def __init__(self, prediction_config, prediction_utils):
        self.prediction_config = prediction_config
        self.prediction_utils = prediction_utils
        self.df_listings = self.load_dataset(None) # Load data set DataFrame (i.e. `None` for all 3723 rows)
        self.training_part = None # Training part of df_listings
        self.testing_part = None # Testing part of df_listings
        self.cleanse_columns() # Data Cleaning, including removal of `?` from Training Features Combination and Target Cols
        self.convert_columns() # Convert to Float64 if String all Training Features Combination and Target Cols
        self.remove_columns_incorrect_format() # Remove columns with inadequate data (i.e. missing values, non-numeric, non-ordinal, unspecific)
        self.retain_columns_low_incomplete_but_strip() # Remove rows with NaN values for Columns where small quantity of them
        self.delete_columns_high_incomplete() # Delete entire Columns with large quantity of NaN rows
        # self.show_columns_incomplete() # Identify quantity of null values per column
        self.normalise_listings()
        self.randomise_listings()
        self.partition_listings()

    def load_dataset(self, num_rows):
        """ Load downloaded copy of dataset (.csv format) into Pandas Dataframe (DF)
        otherwise load directly from remote endpoint (slower)
        """
        try:
            dataset_choice = self.prediction_config.DATASET_CHOICE
            dataset_location_local = self.prediction_config.DATASET_LOCATION[dataset_choice]["local"]
            dataset_location_remote = self.prediction_config.DATASET_LOCATION[dataset_choice]["remote"]
            dataset_file = Path(dataset_location_local)
            if dataset_file.is_file():
                df = pd.read_csv(dataset_location_local, nrows=num_rows)
                return self.add_missing_labels(df)
            else:
                def exists(path):
                    r = requests.head(path)
                    return r.status_code == requests.codes.ok
                if exists(dataset_location_remote):
                    df = pd.read_csv(dataset_location_remote, nrows=num_rows)
                    return self.add_missing_labels(df)
            return None
        except Exception as e:
            print(e.errno)

    def add_missing_labels(self, df):
        """ Check and add labels to top row if necessary """
        dataset_choice = self.prediction_config.DATASET_CHOICE
        dataset_labels = self.prediction_config.DATASET_LOCATION[dataset_choice]["labels"]
        if dataset_labels != "":
            # Convert comma separated labels to list of strings
            df.columns = [str(label) for label in dataset_labels.split(',') if label]
        return df

    def remove_columns_incorrect_format(self):
        """
        Return new object with labels in requested axis removed
        (i.e. axis=1 asks Pandas to drop across DataFrame columns)
        """
        dataset_choice = self.prediction_config.DATASET_CHOICE

        if dataset_choice == "rental-property-listings":
            remove_non_numeric_columns = ["room_type", "city", "state"]
            remove_non_ordinal_columns = ["latitude", "longitude", "zipcode"]
            remove_out_of_scope_columns = ["host_response_rate", "host_acceptance_rate", "host_listings_count"]
            remove_columns = remove_non_numeric_columns + \
                             remove_non_ordinal_columns + \
                             remove_out_of_scope_columns
            self.df_listings.drop(remove_columns, axis=1, inplace=True)

        if dataset_choice == "car-listings":
            remove_non_numeric_columns = ["make", "fuel-type", "aspiration", "body-style", "drive-wheels", "engine-location", "engine-type", "fuel-system"]
            remove_non_ordinal_columns = []
            remove_out_of_scope_columns = ["symboling", "normalized-losses"]
            remove_columns = remove_non_numeric_columns + \
                             remove_non_ordinal_columns + \
                             remove_out_of_scope_columns
            self.df_listings.drop(remove_columns, axis=1, inplace=True)

    # def show_columns_incomplete(self):
    #     """
    #     Show quantity of non-null values for each column for inspection.
    #     Determine columns to remove from the DataFrame (i.e. few non-null)
    #     """
    #     df_size = len(self.df_listings)
    #
    #     # Randomise (not Sorted)
    #     _temp_df_listings_randomised = self.prediction_utils.randomise_dataframe_rows(self.df_listings)
    #
    #     # print("Length of DataFrame: %r" % (df_size))
    #     # print("Prediction Data quantity of non-null data per column: %r" % (_temp_df_listings_randomised.head(n=df_size).info(verbose=True, null_counts=True)))
    #
    #     df_listings_with_any_null_values = _temp_df_listings_randomised[_temp_df_listings_randomised.columns[_temp_df_listings_randomised.isnull().any()].tolist()]
    #
    #     # print("Prediction Data proportion of null data per column for only columns with any null or NaN values: %r" % (self.prediction_utils.get_percentage_missing(df_listings_with_any_null_values)))

    def delete_columns_high_incomplete(self):
        """ Delete Columns where percentage of null or NaN values exceeds MAX_MAJOR_INCOMPLETE value
        These columns may be useless since too many observations missing
        """

        # Iterate over columns in DataFrame
        for name, values in self.df_listings.iteritems():
            # print("%r: %r" % (name, values) )
            if name != "id":
                col_percentage_missing = self.prediction_utils.get_percentage_missing(self.df_listings, name)
                if col_percentage_missing > self.prediction_config.MAX_MAJOR_INCOMPLETE:
                    print("Deleting Column %r, as contains too many null values: %r" % (name, col_percentage_missing) )
                    self.df_listings.drop(name, axis=1, inplace=True)

    def retain_columns_low_incomplete_but_strip(self):
        """ Retain Columns where percentage of null or NaN values comprise LESS THAN 1% (0.01) of its rows
        However remove null and NaN rows along requested axis (i.e. axis=0 asks Pandas to drop across DataFrame rows)
        BEWARE: Do not do this for columns where null/NaN values comprise MORE THAN 1% since
        each rows that is removed from a Column represents an Observation, which is shared across the same
        row of all other Columns in the dataset, so that same row is removed across ALL the Columns
        """

        # For reference only
        dataset_choice = self.prediction_config.DATASET_CHOICE
        training_columns = self.prediction_config.DATASET_LOCATION[dataset_choice]["training_columns"]
        target_column = self.prediction_config.DATASET_LOCATION[dataset_choice]["target_column"]

        # Remove NaN/null values from Columns where percentage of missing values is less than MAX_MINOR_INCOMPLETE
        # and from Columns that are one of the Training set columns or the Target Column
        # Iterate over columns in DataFrame
        new_dc_listings = self.df_listings
        for name, values in self.df_listings.iteritems():
            col_percentage_missing = self.prediction_utils.get_percentage_missing(self.df_listings, name)

            if (col_percentage_missing < self.prediction_config.MAX_MINOR_INCOMPLETE) or (name in training_columns) or (name == target_column):
                # print("Before null/NaN values?: %r" %(self.df_listings[name].isnull().any(axis=0)))
                print("Retained Column: %r, but removed null and NaN valued rows comprising approx. percentage: %r" % (name, col_percentage_missing) )
                new_dc_listings.dropna(axis=0, how="any", subset=[name], inplace=True)
                # print("After null/NaN values?: %r" %(self.df_listings[name].isnull().any(axis=0)))
        # Reindex the DataFrame since some indexes removed (may cause error when iterating later)
        new_dc_listings.reset_index(drop=True, inplace=True)
        self.df_listings = new_dc_listings

    def normalise_listings(self):
        """ Normalise column values where the column types are normalisable, being of either type int, float64, or floating

        BEWARE - Columns containing String values are deleted!

        Apply mass Column transformation to Normalise all the feature columns
        in the df_listings DataFrame and assign to a new Dataframe containing just the
        normalised feature columns to normalized_listings.

        Avoid normalizing the "target_column"
        """

        # Select only Columns containing type int, float64, floating. Exclude Columns with types Object (O) that includes strings
        df_listings_with_float_or_int_values = self.df_listings.select_dtypes(include=['int', 'float64', 'floating'], exclude=['O'])

        normalized_listings = self.prediction_utils.normalise_dataframe(df_listings_with_float_or_int_values)

        dataset_choice = self.prediction_config.DATASET_CHOICE
        target_column = self.prediction_config.DATASET_LOCATION[dataset_choice]["target_column"]
        normalized_listings[target_column] = self.df_listings[target_column]

        print("Normalised listings completed: %r" % (normalized_listings.head(3)) )

        self.df_listings = normalized_listings

    def convert_columns(self):
        """ Convert all Training and Target columns to Numeric type so not removed during normalisation
        and so may be processed by Scikit-Learn
        """

        dataset_choice = self.prediction_config.DATASET_CHOICE
        training_columns = self.prediction_config.DATASET_LOCATION[dataset_choice]["training_columns"]
        target_column = self.prediction_config.DATASET_LOCATION[dataset_choice]["target_column"]

        def convert_to_numeric_type_training_and_target_columns():
            df = self.df_listings
            training_and_target_columns_df = df.filter(training_columns, axis=1)
            for col in training_and_target_columns_df.columns:
                # print("Before Column %r has dtype: %r" % (col, training_and_target_columns_df[col].dtype) )

                # Convert multiple columns using `apply` to the
                # Numeric type. Use `errors` keyword argument to
                # coerce/force not-numeric values to be NaN
                training_and_target_columns_df[col] = pd.to_numeric(training_and_target_columns_df[col], errors='coerce')
                # print("After Column %r has dtype: %r" % (col, training_and_target_columns_df[col].dtype) )

                # Replace old column with new column containing Numeric type
                self.df_listings[col] = training_and_target_columns_df[col]

        convert_to_numeric_type_training_and_target_columns()

    def cleanse_columns(self):
        """ Cleanse all identified price columns.
        Remove ? values from Columns that are one of the Training set columns or the Target Column"""

        dataset_choice = self.prediction_config.DATASET_CHOICE
        training_columns = self.prediction_config.DATASET_LOCATION[dataset_choice]["training_columns"]
        target_column = self.prediction_config.DATASET_LOCATION[dataset_choice]["target_column"]

        def clean_question_marks_from_training_and_target_columns():
            new_dc_listings = self.df_listings
            # Remove ? values from Columns that are one of the Training set columns or the Target Column
            # Iterate over columns in DataFrame
            for name, values in self.df_listings.iteritems():
                if (name in training_columns) or (name == target_column):
                    # Remove rows where the value is "?"
                    df = new_dc_listings
                    to_drop = ['?']
                    print(~df[name].isin(to_drop))
                    new_dc_listings = df[~df[name].isin(to_drop)]
            # Reindex the DataFrame since some indexes removed (may cause error when iterating later)
            new_dc_listings.reset_index(drop=True, inplace=True)
            self.df_listings = new_dc_listings

        def clean_columns_with_price_format():
            for index, price_format_column in enumerate(self.prediction_config.DATASET_LOCATION[dataset_choice]["cleanse_columns_price_format"]):
                if price_format_column in self.df_listings:
                    self.df_listings[price_format_column] = self.prediction_utils.clean_price_format(self.df_listings[price_format_column])

        clean_question_marks_from_training_and_target_columns()
        clean_columns_with_price_format()


    def randomise_listings(self):
        """ Shuffle the ordering of the rows """
        # TODO - Check if duplicate effort when implement KFold as it offers randomisation option
        self.prediction_utils.randomise_dataframe_rows(self.df_listings)

    def partition_listings(self):
        """ Split into partitions using configured technique """

        # Train/Test Validation Process - Splits DataFrame into 2x partitions
        if self.prediction_config.K_FOLD_CROSS_VALIDATION == False:
            training_part_end = self.get_training_partitions(self.df_listings)
            try:
                self.training_part = self.df_listings.iloc[0:training_part_end]
                self.testing_part = self.df_listings.iloc[training_part_end:]
            except Exception as e:
                print(e.errno)
        # K-Fold Cross-Validation Process
        else:
            if self.prediction_config.K_FOLDS_BUILTIN == False:
                self.generate_k_folds_column()
                print("DF Listings 'fold' column %r: " % (self.df_listings["fold"]))

    def generate_k_folds_column(self):
        folds = build_folds(self.df_listings, self.prediction_config.K_FOLDS)
        for index, fold in enumerate(folds):
            # fold_index_from = -1 + fold_number]
            # fold_index_to = fold[0 + fold_number]
            fold_number = index + 1
            fold_index_from = folds[index]
            fold_index_to = folds[fold_number]
            self.df_listings.set_value(self.df_listings.index[fold_index_from:fold_index_to], "fold", int(fold_number))
            if fold_number == (len(folds) - 1):
                break

    def get_training_partitions(self, df):
        """ Split full dataset into Training and Testing sets (DataFrame partition size proportions) """
        testing_proportion = self.prediction_config.TESTING_PROPORTION
        training_len = int(len(df) - len(df) * testing_proportion) # remaining percentage
        # Cater for test_proportion of 0 to prevent out of bounds exception when later increment
        if training_len >= len(df):
            training_len -= len(df) - 1
        return training_len

    def get_training_columns(self):
        """ Return array of Training Columns.

        When "training_columns" array is empty it means return all columns except the "target_column"
        """
        dataset_choice = self.prediction_config.DATASET_CHOICE
        training_columns = self.prediction_config.DATASET_LOCATION[dataset_choice]["training_columns"]
        target_column = self.prediction_config.DATASET_LOCATION[dataset_choice]["target_column"]
        if not training_columns:
            features = self.df_listings.columns.tolist()

            # Remove "target_column"
            features.remove(target_column)

            # Remove columns containing Excluded full text
            for index, column_name in enumerate(self.prediction_config.EXCLUDE_TRAINING_COLUMNS_WITH_FULL_TEXT):
                if column_name in features:
                    features.remove(column_name)

            # Retain columns that do not contain Excluded partial text
            features_to_retain = []
            for index, column_partial_name in enumerate(self.prediction_config.EXCLUDE_TRAINING_COLUMNS_WITH_PARTIAL_TEXT):
                for index, column_name in enumerate(features):
                    if column_partial_name not in column_name:
                        features_to_retain.append(column_name)
            return features_to_retain
        else:
            return training_columns