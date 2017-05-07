import numpy as np

class PredictionConfig():
    """ Machine Learning configuration """

    # Implementation Guide:
    #
    # Predict the optimum value for a future TARGET_COLUMN value based on TRAINING_COLUMNS
    # of a given dataset hosted either remotely DATASET_REMOTE or downloaded locally
    # DATASET_LOCAL.
    #
    #    - Manually add a datasets in CSV format local and remote locations to DATASET_LOCATION,
    #      and add the key of the one to use to DATASET_CHOICE. If the top row of the
    #      dataset does not already include the Label of each comma separated column, then
    #      manually add the respective comma separated Labels in the correct order into
    #      DATASET_LOCATION.labels
    #    - Manually cleanse relevant columns by converting from string to float and removing
    #      bad characters so they may be processed as inputs by the ML Model (i.e. $, ?).
    #      Since there may be a list of columns with the same bad characters (i.e. price)
    #      they may be added to groups such as "cleanse_columns_price_format".
    #      All Training and Target Columns must be cleansed since Scikit-Learn only accepts numeric data
    #    - Manually convert from String to Float64 all Training Features and Target Column
    #    - Automatically cleanse the dataset by removing any columns containing
    #      a string or substring indicative of incorrect data (i.e. id or _id) as defined by
    #      EXCLUDE_TRAINING_COLUMNS_WITH_FULL_TEXT and EXCLUDE_TRAINING_COLUMNS_WITH_PARTIAL_TEXT
    #    - Manually remove entire columns (features) that are non-numeric,
    #      non-ordinal, or not specific to the prediction
    #    - Automatically remove entire columns whose proportion of row
    #      values (observations) that are NaN (null) is > MAX_MAJOR_INCOMPLETE
    #    - Automatically retain columns whose proportion of values that are NaN
    #      is < MAX_MINOR_INCOMPLETE, but remove all rows with NaN value
    #      Important Note: If NaN values are found to cause errors and prevent processing,
    #      check the percentage of NaN rows in the columns chosen to being trained and increase the
    #      value of MAX_MINOR_INCOMPLETE above that percentage
    #    - Reindex after removing rows to prevent errors subsequent in subsequent loops
    #    - Automatically normalise the columns containing int, float64, and floating type values
    #      using mass transformation across the DataFrame to avoid experiencing the Outsized Effect
    #      when applying the Euclidean Distance equation to largely differing values.
    #      Important Note: Currently the normalise function deletes any columns that contain string values
    #      so ensure that this does not occur to Training Features or Target Column by converting to Int or Float prior.
    #    - Manually configure whether to use either the:
    #        - Train/Test Validation Process that splits the dataset into two partitions
    #          Train/Test sets with 75%/25% proportion of rows respectively
    #          by setting K_FOLD_CROSS_VALIDATION to False. The Training set is
    #          used to make predictions whilst the Testing set is used to predict values for.
    #          The Testing percentage proportion is set by TESTING_PROPORTION.
    #        - K-Fold Cross Validation by setting K_FOLD_CROSS_VALIDATION to True
    #          to take return more robust results by rotating through different subsets of
    #          the data to avoid issues of Train/Test Validation by
    #          setting K_FOLDS to a value >= 2
    #    - Manually toggle HYPERPARAMETER_OPTIMISATION to True to try different values of 'k' nearest
    #      neighbors to see comparison plot and find optimum combination of Training set features resulting in lowest MSE
    #      by setting associated range of 'k' values HYPERPARAMETER_RANGE, or else set to
    #      False to and defined a fixed 'k' value with HYPERPARAMETER_FIXED
    #    - Automatically perform prediction and output list of predictions by providing the TRAINING_COLUMNS (all columns
    #      used if this list is empty) and the TARGET_COLUMN's values to the chosen Machine Learning Model
    #      ML_MODEL_KNN (with a "manual" value to use the model built manually or optionally
    #      with a value "external" to use the much faster Scikit-Learn Machine Learning library).
    #      Note that rows of the TRAINING_COLUMNS are used to predict TARGET_COLUMN's values in the Test set.
    #      Note that if only one TRAINING_COLUMN is used it is known as Univariate, whilst more is Multivariate.
    #      The ML Model uses Similarity Metrics (by means of the K-Nearest-Neighbors Machine Learning
    #      algorithm) to iteratively compare columns (features) of two rows (observations) to
    #      calculate (i.e. using Euclidean Distance equation) and return the distance (difference).
    #    - Manually use Error Metrics (i.e. Median Average Error, Mean Squared Error, and
    #      Root Mean Squared Error equations) to check the quality of the predictions.
    #      Increasing the quantity of relevant TRAINING_COLUMNS (attributes) improves accuracy and lowers the error of
    #      the model since it allows the model to better identify observations (rows) from the Training set that are
    #      most similar to the Test set.

    def __init__(self):
        self.ML_MODEL_KNN = "external" # manual or external
        self.HYPERPARAMETER_OPTIMISATION = True # Toggle to iterate through defined HYPERPARAMETER_RANGE of k values
        self.HYPERPARAMETER_RANGE = np.arange(1, 21, 1) # 1 to 20
        self.MIN_FEATURES_COMBO_LEN = 3
        self.HYPERPARAMETER_FIXED = 5 # Fixed value of hyperparameter k when HYPERPARAMETER_OPTIMISATION is False
        self.MAX_MAJOR_INCOMPLETE = 0.2 # Percentage
        self.MAX_MINOR_INCOMPLETE = 0.02 # Percentage
        self.K_FOLD_CROSS_VALIDATION = True
        # K-Fold Cross-Validation Technique when K_FOLDS >= 3 OR Holdout Validation when K_FOLDS == 2
        # Train/Test Validation Process
        self.K_FOLDS = 10
        # Toggle whether to use use:
        #   - True - Scikit-Learn's KFold class to generate K Folds and its cross_val_score function
        #            for training and Cross Validation (without the need to use the "fold" column manually)
        #   - False - Manually generate a KFolds 'fold' column and manually perform Cross Validation
        self.K_FOLDS_BUILTIN = True
        self.TESTING_PROPORTION = self.get_testing_proportion()
        self.DATASET_CHOICE = "car-listings" # "rental-property-listings", "car-listings"
        self.DATASET_LOCATION = {
            "rental-property-listings": {
                "local": "data/listings.csv",
                "remote": "http://data.insideairbnb.com/united-states/dc/washington-dc/2015-10-03/data/listings.csv",
                "labels": "", # Empty array means labels already included in dataset
                # Important Note: empty array means use all as Training Columns except the Target Column
                # Example: # ["accommodates", "bedrooms", "bathrooms", "number_of_reviews"]
                # 3 columns Minimum
                "training_columns": ["accommodates", "bedrooms", "bathrooms", "number_of_reviews"],
                "target_column": "price",
                "cleanse_columns_price_format": ["price"]
            },
            # Automobile Dataset - https://archive.ics.uci.edu/ml/datasets/Automobile
            "car-listings": {
                "local": "data/imports-85.data",
                "remote": "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
                "labels": "symboling,normalized-losses,make,fuel-type,aspiration,num-of-doors,body-style,drive-wheels,engine-location,wheel-base,length,width,height,curb-weight,engine-type,num-of-cylinders,engine-size,fuel-system,bore,stroke,compression-ratio,horsepower,peak-rpm,city-mpg,highway-mpg,price",
                # 3 columns Minimum, Must be Numeric values or converted!
                # i.e. ["wheel-base", "length", "width", "height", "curb-weight", "engine-size", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg"],
                "training_columns": ["height", "curb-weight", "horsepower", "city-mpg", "highway-mpg"],
                "target_column": "price",
                "cleanse_columns_price_format": ["price"]
            }
        }
        self.EXCLUDE_TRAINING_COLUMNS_WITH_FULL_TEXT = ["id"]
        self.EXCLUDE_TRAINING_COLUMNS_WITH_PARTIAL_TEXT = ["_id", "-id"]
        self.validate_config()

    def validate_config(self):
        """ Override config settings in case user has not configured valid combination """

        # Since "manual" mode needs split into Train/Test sets (i.e. training_part and test_part) rather than just a "fold" column
        if self.ML_MODEL_KNN == "manual":
            self.K_FOLD_CROSS_VALIDATION = False

        # Since only implemented K-Fold Cross Validation to work with the Hyperparameter Optimisation process
        if self.K_FOLD_CROSS_VALIDATION == True:
            self.HYPERPARAMETER_OPTIMISATION = True

        # Check that user has assigned the minimum number of features
        dataset_choice = self.DATASET_CHOICE
        training_columns = self.DATASET_LOCATION[dataset_choice]["training_columns"]
        if len(training_columns) < self.MIN_FEATURES_COMBO_LEN:
            print("Error: Configuration requires minimum amount of features assigned to satisfy MIN_FEATURES_COMBO_LEN")

    def get_testing_proportion(self):
        """ Proportion of rows to split into Training and Test set respectively
        Returns value between 0 and 1 representing percentage proportion of the Test set. Training set is remainder.
        Holdout Validation - 50%/50%
        Train/Test Validation - 75%/25%
        """
        if self.K_FOLD_CROSS_VALIDATION:
            return round((1 / self.K_FOLDS), 2)
        else:
            return 0.25