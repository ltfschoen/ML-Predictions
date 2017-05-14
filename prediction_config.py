import numpy as np

class PredictionConfig(object):
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
    #      with a value "scikit" to use the much faster Scikit-Learn Machine Learning library).
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

    def __init__(self, event, context):
        self.CONTEXT = context
        self.ML_MODEL_KNN = event["model_workflow_config"]["model_workflow_for_knn_algorithm"] # manual or scikit
        self.ML_MODEL_LINEAR = event["model_workflow_config"]["model_workflow_for_linear_regression_algorithm_toggle"] # scikit True/False
        self.ML_MODEL_LOGISTIC = event["model_workflow_config"]["model_workflow_for_logistic_regression_algorithm_toggle"]
        self.PLOT_INDIVIDUAL_TRAIN_FEATURES_VS_TARGET = event["plot_config"]["plot_individual_train_features_vs_target_toggle"]
        self.PLOT_LINEAR_RELATIONSHIP_PREDICTION_VS_ACTUAL_FOR_TRAIN_FEATURES_VS_TARGET = event["plot_config"]["plot_linear_relationship_prediction_vs_actual_for_train_features_vs_target_toggle"]
        self.PLOT_LOGISTIC_RELATIONSHIP_PREDICTION_VS_ACTUAL_FOR_TRAIN_FEATURES_VS_TARGET = event["plot_config"]["plot_logistic_relationship_prediction_vs_actual_for_train_features_vs_target_toggle"]
        self.PLOT_LOGISTIC_ROC = event["plot_config"]["plot_logistic_roc"]
        self.PLOT_HYPERPARAMETER_OPTIMISATION = event["plot_config"]["plot_hyperparameter_optimisation"]
        self.HYPERPARAMETER_OPTIMISATION = event["hyperparameter_optimisation_config"]["hyperparameter_optimisation_toggle"] # Toggle to iterate through defined HYPERPARAMETER_RANGE of k values
        self.HYPERPARAMETER_RANGE = np.arange(1, int(event["hyperparameter_optimisation_config"]["hyperparameter_range"]) + 1, 1) # 1 to 20
        self.HYPERPARAMETER_FIXED = int(event["hyperparameter_optimisation_config"]["hyperparameter_quantity_fixed"]) # Fixed value of hyperparameter k when HYPERPARAMETER_OPTIMISATION is False
        self.MIN_FEATURES_COMBO_LEN = int(event["training_config"]["min_training_features"])
        self.MAX_MAJOR_INCOMPLETE = float(event["cleansing_config"]["min_percentage_incomplete_observations_to_remove_column"]) # Percentage
        self.MAX_MINOR_INCOMPLETE = float(event["cleansing_config"]["max_percentage_incomplete_observations_to_retain_column_and_remove_incomplete_slice"]) # Percentage
        self.K_FOLD_CROSS_VALIDATION = event["k_fold_cross_validation_config"]["k_fold_cross_validation_toggle"]
        # K-Fold Cross-Validation Technique when K_FOLDS >= 3 OR Holdout Validation when K_FOLDS == 2
        # Train/Test Validation Process
        self.K_FOLDS = int(event["k_fold_cross_validation_config"]["k_folds_quantity"])
        # Toggle whether to use use:
        #   - True - Scikit-Learn's KFold class to generate K Folds and its cross_val_score function
        #            for training and Cross Validation (without the need to use the "fold" column manually)
        #   - False - Manually generate a KFolds 'fold' column and manually perform Cross Validation
        self.K_FOLDS_BUILTIN = event["k_fold_cross_validation_config"]["k_folds_workflow"] # i.e. manual or scikit
        self.TESTING_PROPORTION = self.get_testing_proportion()
        self.DATASET_CHOICE = event["dataset_selected"] # "rental-property-listings", "car-listings"
        # Important Notes:
        # - "labels" - Use empty array when labels already included in dataset
        # - "training_columns" - Use empty array to use all as Training Columns except the Target Column
        #   Example for "rental-property-listings" dataset: # ["accommodates", "bedrooms", "bathrooms", "number_of_reviews"]
        #   Example for "car-listings" dataset: ["wheel-base", "length", "width", "height", "curb-weight", "engine-size", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg"],
        #   Minimum amount of elements to add to this list is the value of "min_training_features"
        #   Each row value associated with element column in dataset must be numeric or converted prior to usage
        # - "remote" - URL to data associated with dataset. Not the URL to information about the dataset
        #   (i.e. Automobile Dataset information - https://archive.ics.uci.edu/ml/datasets/Automobile)
        # - "convert_columns_words_to_digits" - i.e. convert rows from say "four" to 4
        self.DATASET_LOCATION = event["dataset_config"]
        # TODO - Move into individual dataset config dicts
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