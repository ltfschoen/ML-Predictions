class PredictionConfig(object):
    """ Machine Learning configuration """

    # K-Nearest-Neighbors Machine Learning Model selection:
    #   - external - External (uses Skikit-Learn library) OR
    #   - manual - Manually configured
    ML_MODEL_KNN = "manual" # manual or external
    MAX_MAJOR_INCOMPLETE = 0.2 # Percentage
    MAX_MINOR_INCOMPLETE = 0.02 # Percentage
    TESTING_PROPORTION = 0.25 # Between 0 and 1. i.e. Testing Set 25% of rows. Training Set remaining 75% of rows
    CLEANSE_COLUMNS_PRICE = ["price"]
    DATASET_LOCAL = "data/listings.csv"
    DATASET_REMOTE = "http://data.insideairbnb.com/united-states/dc/washington-dc/2015-10-03/data/listings.csv"
    TRAINING_COLUMNS = ["accommodates", "bedrooms", "bathrooms", "number_of_reviews"]
    TARGET_COLUMN = "price"