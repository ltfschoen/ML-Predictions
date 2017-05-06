import numpy as np

class PredictionConfig():
    """ Machine Learning configuration """

    # Implementation Guide:
    #
    # Predict the optimum value for a future TARGET_COLUMN value based on TRAINING_COLUMNS
    # of a given dataset hosted either remotely DATASET_REMOTE or downloaded locally
    # DATASET_LOCAL.
    #
    #   1) Manually remove entire columns (features) that are non-numeric,
    #      non-ordinal, or not specific to the prediction
    #   2) Automatically remove entire columns whose proportion of row
    #      values (observations) that are NaN (null) is > MAX_MAJOR_INCOMPLETE
    #   3) Automatically retain columns whose proportion of values that are NaN
    #      is < MAX_MINOR_INCOMPLETE, but remove all rows with NaN value
    #      Important Note: If NaN values are found to cause errors and prevent processing,
    #      check the percentage of NaN rows in the columns chosen to being trained and increase the
    #      value of MAX_MINOR_INCOMPLETE above that percentage
    #   4) Manually cleanse relevant columns by converting from string to float and removing
    #      bad characters so they may be processed as inputs by the ML Model (i.e. $).
    #      Since there may be a list of columns with the same bad characters (i.e. price)
    #      they may be added to groups such as CLEANSE_COLUMNS_PRICE
    #   5) Automatically cleanse the dataset by removing any columns containing
    #      a string or substring indicative of incorrect data (i.e. id or _id) as defined by
    #      EXCLUDE_TRAINING_COLUMNS_WITH_FULL_TEXT and EXCLUDE_TRAINING_COLUMNS_WITH_PARTIAL_TEXT
    #   6) Automatically normalise the columns containing int, float64, and floating type values
    #      using mass transformation across the DataFrame to avoid experiencing the Outsized Effect
    #      when applying the Euclidean Distance equation to largely differing values.
    #   7) Manually configure whether to use either the:
    #        - Train/Test Validation Process that splits the dataset into two partitions
    #          Train/Test sets with 75%/25% proportion of rows respectively
    #          by setting K_FOLD_CROSS_VALIDATION to False. The Training set is
    #          used to make predictions whilst the Testing set is used to predict values for.
    #          The Testing percentage proportion is set by TESTING_PROPORTION.
    #        - K-Fold Cross Validation by setting K_FOLD_CROSS_VALIDATION to True
    #          to take return more robust results by rotating through different subsets of
    #          the data to avoid issues of Train/Test Validation by
    #          setting K_FOLDS to a value >= 2
    #   8) Manually toggle HYPERPARAMETER_OPTIMISATION to True to try different values of 'k' nearest
    #      neighbors to see comparison plot and find optimum combination of Training set features resulting in lowest MSE
    #      by setting associated range of 'k' values HYPERPARAMETER_RANGE, or else set to
    #      False to and defined a fixed 'k' value with HYPERPARAMETER_FIXED
    #   9) Automatically perform prediction and output list of predictions by providing the TRAINING_COLUMNS (all columns
    #      used if this list is empty) and the TARGET_COLUMN's values to the chosen Machine Learning Model
    #      ML_MODEL_KNN (with a "manual" value to use the model built manually or optionally
    #      with a value "external" to use the much faster Scikit-Learn Machine Learning library).
    #      Note that rows of the TRAINING_COLUMNS are used to predict TARGET_COLUMN's values in the Test set.
    #      Note that if only one TRAINING_COLUMN is used it is known as Univariate, whilst more is Multivariate.
    #      The ML Model uses Similarity Metrics (by means of the K-Nearest-Neighbors Machine Learning
    #      algorithm) to iteratively compare columns (features) of two rows (observations) to
    #      calculate (i.e. using Euclidean Distance equation) and return the distance (difference).
    #   10) Manually use Error Metrics (i.e. Median Average Error, Mean Squared Error, and
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
        # K-fold Cross-Validation Technique when K_FOLDS >= 3 OR Holdout Validation when K_FOLDS == 2
        # Train/Test Validation Process
        self.K_FOLDS = 10
        self.TESTING_PROPORTION = self.get_testing_proportion()
        self.CLEANSE_COLUMNS_PRICE = ["price"]
        self.DATASET_LOCAL = "data/listings.csv"
        self.DATASET_REMOTE = "http://data.insideairbnb.com/united-states/dc/washington-dc/2015-10-03/data/listings.csv"
        self.EXCLUDE_TRAINING_COLUMNS_WITH_FULL_TEXT = ["id"]
        self.EXCLUDE_TRAINING_COLUMNS_WITH_PARTIAL_TEXT = ["_id", "-id"]
        # Example: # ["accommodates", "bedrooms", "bathrooms", "number_of_reviews"]
        self.TRAINING_COLUMNS = ["accommodates", "bedrooms", "bathrooms", "number_of_reviews"] # Important Note: empty array means use all as Training Columns except the Target Column
        self.TARGET_COLUMN = "price"
        self.validate_config()

    def validate_config(self):
        """ Override config settings in case user has not configured valid combination """

        # Since "manual" mode needs split into Train/Test sets (i.e. training_part and test_part) rather than just a "fold" column
        if self.ML_MODEL_KNN == "manual":
            self.K_FOLD_CROSS_VALIDATION = False

        # Since only implemented K-Fold Cross Validation to work with the Hyperparameter Optimisation process
        if self.K_FOLD_CROSS_VALIDATION == True:
            self.HYPERPARAMETER_OPTIMISATION = True

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