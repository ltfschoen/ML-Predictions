import numpy

class PredictionConfig(object):
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
    #   7) Automatically splitting the dataset it into two partitions (Training set is
    #      used to make predictions whilst the Testing set is used to predict values for)
    #      for the Train/Test Validation Process, where the Testing percentage proportion
    #      is set by TESTING_PROPORTION.
    #   8) Automatically perform prediction and output list of predictions by providing the TRAINING_COLUMNS (all columns
    #      used if this list is empty) and the TARGET_COLUMN's values to the chosen Machine Learning Model
    #      ML_MODEL_KNN (with a "manual" value to use the model built manually or optionally
    #      with a value "external" to use the much faster Scikit-Learn Machine Learning library).
    #      Note that rows of the TRAINING_COLUMNS are used to predict TARGET_COLUMN's values in the Test set.
    #      Note that if only one TRAINING_COLUMN is used it is known as Univariate, whilst more is Multivariate.
    #      The ML Model uses Similarity Metrics (by means of the K-Nearest-Neighbors Machine Learning
    #      algorithm) to iteratively compare columns (features) of two rows (observations) to
    #      calculate (i.e. using Euclidean Distance equation) and return the distance (difference).
    #   9) Manually use Error Metrics (i.e. Median Average Error, Mean Squared Error, and
    #      Root Mean Squared Error equations) to check the quality of the predictions.
    #      Increasing the quantity of relevant TRAINING_COLUMNS (attributes) improves accuracy and lowers the error of
    #      the model since it allows the model to better identify observations (rows) from the Training set that are
    #      most similar to the Test set.

    ML_MODEL_KNN = "external" # manual or external
    HYPERPARAMETER_OPTIMISATION = True # Toggle to iterate through defined HYPERPARAMETER_RANGE of k values
    HYPERPARAMETER_RANGE = numpy.arange(1, 21, 1) # 1 to 20
    HYPERPARAMETER_FIXED = 5 # Fixed value of hyperparameter k when HYPERPARAMETER_OPTIMISATION is False
    MAX_MAJOR_INCOMPLETE = 0.2 # Percentage
    MAX_MINOR_INCOMPLETE = 0.02 # Percentage
    TESTING_PROPORTION = 0.25 # Between 0 and 1. i.e. Testing Set 25% of rows. Training Set remaining 75% of rows
    CLEANSE_COLUMNS_PRICE = ["price"]
    DATASET_LOCAL = "data/listings.csv"
    DATASET_REMOTE = "http://data.insideairbnb.com/united-states/dc/washington-dc/2015-10-03/data/listings.csv"
    EXCLUDE_TRAINING_COLUMNS_WITH_FULL_TEXT = ["id"]
    EXCLUDE_TRAINING_COLUMNS_WITH_PARTIAL_TEXT = ["_id", "-id"]
    # Example: # ["accommodates", "bedrooms", "bathrooms", "number_of_reviews"]
    TRAINING_COLUMNS = [] # Important Note: empty array means use all as Training Columns except the Target Column
    TARGET_COLUMN = "price"