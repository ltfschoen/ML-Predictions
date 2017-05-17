import pandas as pd
from pathlib import Path
import requests
from lib.build_folds import build_folds
import copy
from itertools import compress
import numpy as np
import clustering_model_kmeans_external

class PredictionData:
    """ Load and Partition DataFrame into Training/Testing for Validation Process """
    def __init__(self, prediction_config, prediction_utils):
        self.prediction_config = prediction_config
        self.prediction_utils = prediction_utils
        self.df_listings = self.load_dataset(None) # Load data set DataFrame (i.e. `None` for all 3723 rows)
        self.training_part = None # Training part of df_listings
        self.testing_part = None # Testing part of df_listings
        self.dataset_choice = self.prediction_config.DATASET_CHOICE
        self.target_column = self.prediction_config.DATASET_LOCATION[self.dataset_choice]["target_column"]
        self.training_columns = self.setup_training_columns() # If no training columns specified then all are chosen
        self.multi_classification_input_columns = self.prediction_config.DATASET_LOCATION[self.prediction_config.DATASET_CHOICE]["multi_classification_input_columns"]
        self.categorical_columns = []
        self.validate_config()
        self.cleanse_columns() # Data Cleaning, including removal of `?` from Training Features Combination and Target Cols
        self.convert_columns() # Convert to Float64 if String all Training Features Combination and Target Cols
        self.convert_categorical_columns_to_dummy_binary_columns()
        self.remove_columns_incorrect_format() # Remove columns with inadequate data (i.e. missing values, non-numeric, non-ordinal, unspecific)
        self.retain_columns_low_incomplete_but_strip() # Remove rows with NaN values for Columns where small quantity of them
        self.delete_columns_high_incomplete() # Delete entire Columns with large quantity of NaN rows
        # self.show_columns_incomplete() # Identify quantity of null values per column
        self.k_means_clustering()
        self.delete_columns_low_correlation() # Must occur after K-Means Clustering since Target column (i.e. 'extremism') may not be generated yet
        self.normalise_listings()
        self.randomise_listings()
        self.partition_listings()
        self.validate_training_columns()

    def load_dataset(self, num_rows):
        """ Load downloaded copy of dataset (.csv format) into Pandas Dataframe (DF)
        otherwise load directly from remote endpoint (slower)
        """
        try:
            dataset_choice = self.prediction_config.DATASET_CHOICE
            dataset_location_local = self.prediction_config.DATASET_LOCATION[dataset_choice]["local"]
            dataset_location_remote = self.prediction_config.DATASET_LOCATION[dataset_choice]["remote"]
            dataset_file = Path(dataset_location_local)
            dataset_file_format = self.prediction_config.DATASET_LOCATION[dataset_choice]["format"]
            if dataset_file.is_file():
                if dataset_file_format == "csv-whitespace-separated":
                    df = pd.read_table(dataset_location_local, delim_whitespace=True)
                else:
                    df = pd.read_csv(dataset_location_local, nrows=num_rows)
                return self.add_missing_labels(df)
            else:
                def exists(path):
                    r = requests.head(path)
                    return r.status_code == requests.codes.ok
                if exists(dataset_location_remote):
                    if dataset_file_format == "csv-whitespace-separated":
                        df = pd.read_table(dataset_location_local, delim_whitespace=True)
                    else:
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

    def validate_config(self):
        if isinstance(self.df_listings, type(None)):
            return

        # Ensure that quantity of K-Fold splits is not greater than quantity of samples
        if self.prediction_config.K_FOLDS > len(self.df_listings):
            self.prediction_config.K_FOLDS = len(self.df_listings)

        # Ensure that quantity of neighbors range (of hyperparameter k) is not greater than quantity of samples
        if len(self.prediction_config.HYPERPARAMETER_RANGE) > len(self.df_listings):
            self.prediction_config.HYPERPARAMETER_RANGE = np.arange(1, int(len(self.df_listings)), 1)

    def update_training_columns_with_removed(self, column_name):
        if column_name in self.training_columns:
            self.training_columns.remove(column_name)

    def remove_columns_incorrect_format(self):
        """
        Return new object with labels in requested axis removed
        (i.e. axis=1 asks Pandas to drop across DataFrame columns)
        """
        if isinstance(self.df_listings, type(None)):
            return

        _temp_df_listings = copy.deepcopy(self.df_listings)

        remove_non_numeric_columns = self.prediction_config.DATASET_LOCATION[self.dataset_choice]["exclude_columns"]["non_numeric"]
        remove_non_ordinal_columns = self.prediction_config.DATASET_LOCATION[self.dataset_choice]["exclude_columns"]["non_ordinal"]
        remove_out_of_scope_columns = self.prediction_config.DATASET_LOCATION[self.dataset_choice]["exclude_columns"]["out_of_scope"]
        remove_columns = remove_non_numeric_columns + remove_non_ordinal_columns + remove_out_of_scope_columns

        existing_columns_to_remove = []
        for index, name in enumerate(remove_columns):
            if name in self.df_listings.columns:
                existing_columns_to_remove.append(name)
                self.update_training_columns_with_removed(name)

        _temp_df_listings.drop(existing_columns_to_remove, axis=1, inplace=True, errors='raise')

        # Remove white spaces in column names
        _temp_df_listings.columns = _temp_df_listings.columns.str.strip()

        # Drop range of rows for specified columns
        exclude_columns = self.prediction_config.DATASET_LOCATION[self.dataset_choice]["exclude_columns"]
        if "remove_range" in exclude_columns:
            remove_rows_for_columns = self.prediction_config.DATASET_LOCATION[self.dataset_choice]["exclude_columns"]["remove_range"]

            print("DataFrame row count before filtered row removal: ", _temp_df_listings.shape[0])
            if len(remove_rows_for_columns.keys()) != 0:
                for column_name in remove_rows_for_columns:
                    if column_name in self.df_listings.columns:
                        if len(remove_rows_for_columns[column_name].keys()) != 0:
                            for range_string in remove_rows_for_columns[column_name]:

                                def get_rows_to_drop_indexes(_rows_to_drop):
                                    _original_df_indexes_to_drop = _rows_to_drop.index.values.tolist()
                                    return list(range(0, len(_original_df_indexes_to_drop)))

                                if range_string == "lteq":
                                    # Pass second optional argument value to .get to return it as default if no value exists
                                    _rows_to_drop = _temp_df_listings[self.df_listings[column_name] <= remove_rows_for_columns[column_name].get('lteq', 0)]
                                    _temp_df_listings.drop(_rows_to_drop.index[get_rows_to_drop_indexes(_rows_to_drop)], axis=0, inplace=True, errors='ignore')
                                elif range_string == "gt":
                                    _rows_to_drop = _temp_df_listings[self.df_listings[column_name] > remove_rows_for_columns[column_name].get('gt', 0)]
                                    _temp_df_listings.drop(_rows_to_drop.index[get_rows_to_drop_indexes(_rows_to_drop)], axis=0, inplace=True, errors='ignore')
            print("DataFrame row count after filtered row removal: ", _temp_df_listings.shape[0])
        self.df_listings = _temp_df_listings

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
        print("Deleting Columns with high incomplete...")

        # Iterate over columns in DataFrame
        if isinstance(self.df_listings, type(None)):
            return

        for name, values in self.df_listings.iteritems():
            # print("%r: %r" % (name, values) )
            col_percentage_missing = self.prediction_utils.get_percentage_missing(self.df_listings, name)

            if col_percentage_missing > self.prediction_config.MAX_MAJOR_INCOMPLETE:
                print("Deleting Column %r, as contains too many null values: %r" % (name, col_percentage_missing) )
                self.df_listings.drop(name, axis=1, inplace=True)
                self.update_training_columns_with_removed(name)

    def delete_columns_low_correlation(self):
        """ Find pairwise Correlations with Target Column to identify columns to remove:
          - Identify Columns that do not correlate and add predictive power to the model
          - Identify Columns to remove that are derived from the Target Column to avoid overfitting
        """
        print("Deleting Columns with low correlation to Target Column...")

        # Iterate over columns in DataFrame
        if isinstance(self.df_listings, type(None)):
            return

        correlations = self.df_listings.corr()

        self.prediction_utils.plot_corr(self.df_listings)
        # print(correlations[target_column])

        corr_map = {}
        for dict_index, dict_key in enumerate(correlations):
            if dict_key != self.target_column:
                corr_map[dict_index] = dict_key

        # Remove Columns less than % Correlated to Target Column
        # correlations[self.target_column].pop(self.target_column) # Remove Target Column as not need correlate with itself
        for other_column_index, corr_value in enumerate(correlations[self.target_column]):
            min_corr = self.prediction_config.MIN_PERCENTAGE_CORRELATION_WITH_TARGET_COLUMN
            if corr_value < min_corr:
                if corr_map[other_column_index] != self.target_column:
                    print("Deleting Column %r, as its correlation of %r is lower than minimum required of %r" % (corr_map[other_column_index], corr_value, min_corr))
                    self.df_listings.drop(corr_map[other_column_index], axis=1, inplace=True)
                    self.update_training_columns_with_removed(corr_map[other_column_index])
        # Reindex the DataFrame since some indexes removed (may cause error when iterating later)
        self.df_listings.reset_index(drop=True, inplace=True)

    def retain_columns_low_incomplete_but_strip(self):
        """ Retain Columns where percentage of null or NaN values comprise LESS THAN 1% (0.01) of its rows
        However remove null and NaN rows along requested axis (i.e. axis=0 asks Pandas to drop across DataFrame rows)
        BEWARE: Do not do this for columns where null/NaN values comprise MORE THAN 1% since
        each rows that is removed from a Column represents an Observation, which is shared across the same
        row of all other Columns in the dataset, so that same row is removed across ALL the Columns
        """
        print("Retaining columns with low incomplete but stripping...")

        # Remove NaN/null values from Columns where percentage of missing values is less than MAX_MINOR_INCOMPLETE
        # and from Columns that are one of the Training set columns or the Target Column
        # Iterate over columns in DataFrame
        if isinstance(self.df_listings, type(None)):
            return

        new_dc_listings = self.df_listings
        for name, values in self.df_listings.iteritems():
            col_percentage_missing = self.prediction_utils.get_percentage_missing(self.df_listings, name)

            # Important to only apply to Training and Target Columns but not all, otherwise expect to potentially get error:
            # `ValueError: Expected n_neighbors <= n_samples`
            if (col_percentage_missing < self.prediction_config.MAX_MINOR_INCOMPLETE) and ((name in self.training_columns) or (name == self.target_column)):
                # print("Before null/NaN values?: %r" %(self.df_listings[name].isnull().any(axis=0)))
                print("Retained Column: %r, but removed null and NaN valued rows comprising approx. percentage: %r" % (name, col_percentage_missing) )
                new_dc_listings.dropna(axis=0, how="any", subset=[name], inplace=True)
                # print("After null/NaN values?: %r" %(self.df_listings[name].isnull().any(axis=0)))
        # Reindex the DataFrame since some indexes removed (may cause error when iterating later)
        new_dc_listings.reset_index(drop=True, inplace=True)
        self.df_listings = new_dc_listings

    def k_means_clustering(self):
        if self.prediction_config.K_MEANS_CLUSTERING == True:
            clustering_model_kmeans_external.run(self.prediction_config, self.df_listings, self.prediction_utils)

    def normalise_listings(self):
        """ Normalise column values where the column types are normalisable, being of either type int, float64, or floating

        BEWARE - Columns containing String values are deleted!

        Apply mass Column transformation to Normalise all the feature columns
        in the df_listings DataFrame and assign to a new Dataframe containing just the
        normalised feature columns to normalized_listings.

        Avoid normalizing the "target_column"
        """
        print("Normalising...")

        # Select only Columns containing type int, float64, floating. Exclude Columns with types Object (O) that includes strings
        if isinstance(self.df_listings, type(None)):
            return

        # Exclude categorical_columns from Normalisation
        df_exclude_categorical_cols = self.df_listings[self.df_listings.columns.difference(self.categorical_columns)]

        df_listings_with_float_or_int_values = df_exclude_categorical_cols.select_dtypes(include=['int', 'int64', 'float64', 'floating'], exclude=['O'])
        print("Excluding non-numeric columns and categorical columns from Normalisation: ")

        excluding_columns = self.df_listings.select_dtypes(include=['O']).columns.tolist()
        for index, name in enumerate(excluding_columns):
            self.update_training_columns_with_removed(name)

        normalized_listings = self.prediction_utils.normalise_dataframe(df_listings_with_float_or_int_values)

        dataset_choice = self.prediction_config.DATASET_CHOICE
        target_column = self.prediction_config.DATASET_LOCATION[dataset_choice]["target_column"]
        normalized_listings[target_column] = self.df_listings[target_column]

        print("Normalised listings completed: %r" % (normalized_listings.head(3)) )

        # Find Categorical Columns still in the main DataFrame (not cleansed)
        remaining_categorical_columns = []
        df_listings_columns = self.df_listings.columns
        for index, value in enumerate(df_listings_columns):
            if value in self.categorical_columns:
                remaining_categorical_columns.append(value)

        # Update list of remaining categorical columns
        self.categorical_columns = remaining_categorical_columns

        # Concatenate the Categorical Columns back onto Normalised Columns
        df_categorical_columns_remaining = self.df_listings[remaining_categorical_columns]
        normalized_listings_with_categorical_retained = pd.concat([normalized_listings, df_categorical_columns_remaining], axis=1)
        self.df_listings = normalized_listings_with_categorical_retained

    def convert_columns(self):
        """ Convert all Training and Target columns to Numeric type so not removed during normalisation
        and so may be processed by Scikit-Learn
        """
        if isinstance(self.df_listings, type(None)):
            return

        def convert_column_words_to_digits():
            """ Convert rows of specific Columns that have numbers in word string form (i.e. one, three, five, instead of 1, 3, 5) """

            words_for_digits = self.prediction_config.DATASET_LOCATION[self.dataset_choice]["convert_columns_words_to_digits"]
            df = self.df_listings

            if words_for_digits:
                for index, column in enumerate(words_for_digits):
                    df[column] = df[column].map({
                        'one': 1,
                        'two': 2,
                        'three': 3,
                        'four': 4,
                        'five': 5,
                        'six': 6
                    })
            self.df_listings = df

        def convert_to_numeric_type_training_and_target_columns():

            df = self.df_listings
            _training_and_target_columns = copy.deepcopy(self.training_columns)
            _training_and_target_columns.extend([self.target_column])
            _training_and_target_columns_df = df.filter(_training_and_target_columns, axis=1)
            _df_for_new_cols = self.df_listings
            for index, col in enumerate(_training_and_target_columns_df.columns):
                print("Before Column %r has dtype: %r" % (col, _training_and_target_columns_df[col].dtype) )

                # Convert multiple columns using `apply` to the
                # Numeric type. Use `errors` keyword argument to
                # coerce/force not-numeric values to be NaN

                # Do not convert columns with non-numeric string values
                if not col in self.prediction_config.DATASET_LOCATION[self.dataset_choice]["exclude_columns"]["non_numeric"]:
                    _df_for_new_cols[col] = _training_and_target_columns_df[col]
                else:
                    _df_for_new_cols[col] = pd.to_numeric(_training_and_target_columns_df[col], errors='coerce')

                print("After Column %r has dtype: %r" % (col, _df_for_new_cols[col].dtype) )

                # Replace old column with new column containing Numeric type
                self.df_listings[col] = _df_for_new_cols[col]

        convert_column_words_to_digits()
        convert_to_numeric_type_training_and_target_columns()

    def convert_categorical_columns_to_dummy_binary_columns(self):
        """ Convert Categorical Columns (containing multiple categories) to new Dummy Binary Columns """
        if not (len(self.multi_classification_input_columns) > 0):
            return

        def get_dummy_columns():
            df_dummy_columns = pd.DataFrame()
            df_multi_classification_dummy_map = {}
            # Generate and add new Dummy Binary Columns to main DataFrame
            for index, column in enumerate(self.multi_classification_input_columns):
                prefix = column[:3] # Extract first 3 characters
                _temp_df = pd.DataFrame()
                _temp_df = pd.concat([_temp_df, pd.get_dummies(self.df_listings[column], prefix=prefix)], axis=1)
                df_multi_classification_dummy_map[column] = _temp_df.columns
                df_dummy_columns = pd.concat([df_dummy_columns, _temp_df], axis=1)
            return df_dummy_columns, df_multi_classification_dummy_map
        df_dummy_columns, df_multi_classification_dummy_map = get_dummy_columns()
        df_dummy_column_names = df_dummy_columns.columns
        self.categorical_columns += list(df_dummy_column_names)

        # Add new Dummy Binary Columns to main DataFrame
        self.df_listings = pd.concat([self.df_listings, df_dummy_columns], axis=1)

        # Update Training Columns to now have the Dummy Binary Columns if any of the converted Categorical Columns were Training Columns
        for key, value in df_multi_classification_dummy_map.items():
            if key in self.training_columns:
                self.training_columns += list(value)

        # Remove columns from the Training Columns if they were ones that were converted to Dummy Binary Columns
        for index, value in enumerate(self.multi_classification_input_columns):
            if value in self.training_columns:
                self.training_columns.remove(value)

        # Remove Multi-Classification columns from DataFrame that have now been converted to new Dummy Binary Columns
        def remove_converted_columns(col_names):
            self.df_listings.drop(col_names, axis=1, inplace=True, errors='raise')
        remove_converted_columns(self.multi_classification_input_columns)


    def cleanse_columns(self):
        """ Cleanse all identified price columns.
        Remove ? values from Columns that are one of the Training set columns or the Target Column"""

        if isinstance(self.df_listings, type(None)):
            return

        def clean_question_marks_from_training_and_target_columns():
            new_dc_listings = self.df_listings
            # Remove ? values from Columns that are one of the Training set columns or the Target Column
            # Iterate over columns in DataFrame
            for name, values in self.df_listings.iteritems():
                if (name in self.training_columns) or (name == self.target_column):
                    # Remove rows where the value is "?"
                    df = new_dc_listings
                    to_drop = ['?']
                    # print(~df[name].isin(to_drop))
                    new_dc_listings = df[~df[name].isin(to_drop)]
            # Reindex the DataFrame since some indexes removed (may cause error when iterating later)
            new_dc_listings.reset_index(drop=True, inplace=True)
            self.df_listings = new_dc_listings

        def clean_columns_with_price_format():
            for index, price_format_column in enumerate(self.prediction_config.DATASET_LOCATION[self.dataset_choice]["cleanse_columns_price_format"]):
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
            if self.prediction_config.K_FOLDS_BUILTIN == "manual":
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

    def setup_training_columns(self):
        """ Return array of Training Columns.

        When "training_columns" array is empty it means return all columns except the "target_column"
        """

        training_columns = self.prediction_config.DATASET_LOCATION[self.dataset_choice]["training_columns"]

        if not training_columns and not isinstance(self.df_listings, type(None)):
            features = self.df_listings.columns.tolist()

            # Remove "target_column" (if already in the dataset, as may not yet have been generated by Clustering)
            if self.target_column in features:
                features.remove(self.target_column)

            # Remove columns containing Excluded full text
            for index, column_name in enumerate(self.prediction_config.EXCLUDE_TRAINING_COLUMNS_WITH_FULL_TEXT):
                if column_name in features:
                    features.remove(column_name)

            # Retain columns that do not contain Excluded partial text
            is_features_to_retain = [False] * len(features)
            for idx_outer, column_partial_name in enumerate(self.prediction_config.EXCLUDE_TRAINING_COLUMNS_WITH_PARTIAL_TEXT):
                for idx_inner, column_name in enumerate(features):
                    if column_partial_name not in column_name:
                        is_features_to_retain[idx_inner] = True
            filtered = list(compress(features, is_features_to_retain))
            return filtered
        else:
            return training_columns

    def validate_training_columns(self):
        """ Check training columns match columns in dataset after finish setting up data when all training columns used """

        if isinstance(self.df_listings, type(None)):
            return

        training_columns = self.prediction_config.DATASET_LOCATION[self.dataset_choice]["training_columns"]
        print("Warning: Using all dataset columns as Training Columns may take forever!! Try to limit to 4 maximum")
        if not training_columns:
            new_training_columns = self.df_listings.columns.tolist()
            new_training_columns.remove(self.target_column)
            self.training_columns = new_training_columns

        print("Training Columns: ", len(self.training_columns))
        # Check that user has assigned the minimum number of features
        if len(self.training_columns) < self.prediction_config.MIN_FEATURES_COMBO_LEN:
            raise ValueError("MIN_FEATURES_COMBO_LEN not satisfied")

        # Check that if Logistic Regression is enabled that the Target column is Categorical (i.e. int64, not float64)
        if self.df_listings[self.target_column].dtype != "int64" and self.prediction_config.ML_MODEL_LOGISTIC == True:
            raise ValueError("Target column must be Categorical type i.e. int64 NOT float64 when ML_MODEL_LOGISTIC is True")