import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
from prediction_config import PredictionConfig

class PredictionUtils(object):
    """ Utility functions """

    @staticmethod
    def normalise_dataframe(df):
        """ Apply mass Column transformation to Normalise all feature columns in a DataFrame """
        return (df - df.mean()) / (df.std())

    @staticmethod
    def get_percentage_missing(series):
        """ Calculates percentage of NaN values in DataFrame
        :param series: Pandas DataFrame object
        :return: float
        """
        num = series.isnull().sum()
        den = len(series)
        return round(num/den, 2)

    @staticmethod
    def calc_euclidean_dist(val1, val2):
        """ Euclidean Distance equation to compare values of different data sets """
        if np.isnan(val1) or np.isnan(val2):
            return 2**5 # high number so exclude when sort (infinity as integer 2**100000)
        return int(math.sqrt(abs(val1 - val2)**2))

    @staticmethod
    def calc_euclidean_dist_using_scipy(val1, val2):
        """ SciPy distance.euclidean() function used to calculate Euclidean Distance """
        if np.isnan(val1) or np.isnan(val2):
            return 2**5 # high number so exclude when sort (infinity as integer 2**100000)
        return distance.euclidean(val1, val2) # int(math.sqrt(abs(val1 - val2)**2))

    @staticmethod
    def compare_observations(obs1, obs2):
        """ Similarity Metric compares two observations' data set features (columns)
        and returns distance (difference). Compare value of feature
        (i.e. "accommodates" or "bathrooms") in across DataFrame Series
        """
        return obs2.apply(lambda x: PredictionUtils.calc_euclidean_dist_using_scipy(x, obs1))

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
        """ Filter range of nearest neighbors to select of recommended prices (TARGET_COLUMN) to charge per night for a rental listing based
        on average price of other listings based on the model feature being trained
        (i.e. "accommodates" or "bathrooms").
        """
        print("Predicted Target Column (i.e. 'price') (Avg of Nearest): %.2f (with model feature %r Avg. : %r) " % (df.iloc[0:5][PredictionConfig.TARGET_COLUMN].mean(), model_feature_name, df.iloc[0:5][model_feature_name].mean()) )
        return df.iloc[0:5][PredictionConfig.TARGET_COLUMN].mean()

    @staticmethod
    def calc_mean_absolute_error(df, model_feature_name):
        """ MAE = ( |(actual1 - predicted1)| + ... + |(actualn - predictedn)| ) / n """
        column_name_predicted_target = "predicted_" + PredictionConfig.TARGET_COLUMN + "_" + model_feature_name
        return df.apply(lambda x: np.absolute(x[PredictionConfig.TARGET_COLUMN] - x[column_name_predicted_target]), axis=1).mean()

    @staticmethod
    def calc_mean_squared_error(df, model_feature_name):
        """ MSE = ( (actual1 - predicted1)^2 + ... + (actualn - predictedn)^2 ) / n """
        column_name_predicted_target = "predicted_" + PredictionConfig.TARGET_COLUMN + "_" + model_feature_name
        return df.apply(lambda x: (x[PredictionConfig.TARGET_COLUMN] - x[column_name_predicted_target])**2, axis=1).mean()

    @staticmethod
    def calc_root_mean_squared_error(df, model_feature_name):
        """ RMSE = sqrt(MSE) """
        return np.sqrt( PredictionUtils.calc_mean_squared_error(df, model_feature_name) )

    @staticmethod
    def plot(training_model_feature_name, testing_part):
        """ Plot """
        testing_part.pivot_table(index=training_model_feature_name, values=PredictionConfig.TARGET_COLUMN).plot()
        plt.show()