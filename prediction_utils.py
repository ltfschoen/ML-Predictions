import numpy as np
import math

class PredictionUtils(object):
    """ Utility functions """

    @staticmethod
    def calc_euclidean_dist(val1, val2):
        """ Euclidean Distance equation to compare values of different data sets """
        if np.isnan(val1) or np.isnan(val2):
            return 2**5 # high number so exclude when sort (infinity as integer 2**100000)
        return int(math.sqrt(abs(val1 - val2)**2))

    @staticmethod
    def compare_observations(obs1, obs2):
        """ Similarity Metric compares two observations' data set features (columns)
        and returns distance (difference). Compare value of feature
        (i.e. "accommodates" or "bathrooms") in across DataFrame Series
        """
        return obs2.apply(lambda x: PredictionUtils.calc_euclidean_dist(x, obs1))

    @staticmethod
    def randomise_dataframe_rows(df):
        """ Randomise ordering of DataFrame.
        Return a NumPy array of shuffled index values using `np.random.permutation`
        Return a new Dataframe containing the shuffled order using `loc[]`
        `seed(1)` reproduces random same results when share and run same code by others
        """
        np.random.seed(1)
        return df.loc[np.random.permutation(len(df))]

    @staticmethod
    def sort_dataframe_by_feature(df, feature):
        """ Sort DataFrame by feature (default ascending).
        Sort the DataFrame by "distance" column so there will be random order across the
        rows at the top of the list (having same lowest distance).
        """
        return df.sort_values(feature)

    @staticmethod
    def clean_price(df):
        """ Clean "price" column removing `$` and `,` chars. Convert column from text to float. """
        def replace_bad_chars(row):
            row = str(row).replace(",", "")
            row = str(row).replace("$", "")
            row = float(row) # .astype('float')
            return row
        df["price"] = df["price"].apply(lambda row: replace_bad_chars(row))
        return df

    @staticmethod
    def get_nearest_neighbors(df, model_feature_name):
        """ Filter range of nearest neighbors to select of recommended prices to charge per night for a rental listing based
        on average price of other listings based on the model feature being trained
        (i.e. "accommodates" or "bathrooms").
        """
        print("Predicted Price (Avg of Nearest): %.2f (with model feature %r Avg. : %r) " % (df.iloc[0:5]["price"].mean(), model_feature_name, df.iloc[0:5][model_feature_name].mean()) )
        return df.iloc[0:5]["price"].mean()

    @staticmethod
    def calc_mean_absolute_error(df, model_feature_name):
        """ MAE = ( |(actual1 - predicted1)| + ... + |(actualn - predictedn)| ) / n """
        column_name_predicted_price_feature = "predicted_price_" + model_feature_name
        return df.apply(lambda x: np.absolute(x['price'] - x[column_name_predicted_price_feature]), axis=1).mean()

    @staticmethod
    def calc_mean_squared_error(df, model_feature_name):
        """ MSE = ( (actual1 - predicted1)^2 + ... + (actualn - predictedn)^2 ) / n """
        column_name_predicted_price_feature = "predicted_price_" + model_feature_name
        return df.apply(lambda x: (x['price'] - x[column_name_predicted_price_feature])**2, axis=1).mean()

    @staticmethod
    def calc_root_mean_squared_error(df, model_feature_name):
        """ RMSE = sqrt(MSE) """
        return np.sqrt( PredictionUtils.calc_mean_squared_error(df, model_feature_name) )