import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')
from scipy.spatial import distance

class PredictionUtils():
    """ Utility functions """

    def __init__(self, prediction_config):
        self.prediction_config = prediction_config
        self.target_column = self.prediction_config.DATASET_LOCATION[self.prediction_config.DATASET_CHOICE]["target_column"]

    def normalise_dataframe(self, df):
        """ Apply mass Column transformation to Normalise all feature columns in a DataFrame """
        return (df - df.mean()) / (df.std())

    def get_percentage_missing(self, df, column):
        """ Calculates percentage of NaN values in DataFrame
        :param series: Pandas DataFrame object
        :return: float
        """
        count_question_marks = 0
        for i, v in enumerate(df[column]):
            if df[column][i] == "?":
                count_question_marks += 1
        num = df[column].isnull().sum() + count_question_marks
        den = len(df[column])
        return round(num/den, 2)

    def calc_euclidean_dist(self, val1, val2):
        """ Euclidean Distance equation to compare values of different data sets """
        if np.isnan(val1) or np.isnan(val2):
            return 2**5 # high number so exclude when sort (infinity as integer 2**100000)
        return int(math.sqrt(abs(val1 - val2)**2))

    def calc_euclidean_dist_using_scipy(self, val1, val2):
        """ SciPy distance.euclidean() function used to calculate Euclidean Distance """
        if np.isnan(val1) or np.isnan(val2):
            return 2**5 # high number so exclude when sort (infinity as integer 2**100000)
        return distance.euclidean(val1, val2) # int(math.sqrt(abs(val1 - val2)**2))

    def compare_observations(self, obs1, obs2):
        """ Similarity Metric compares two observations' data set features (columns)
        and returns distance (difference). Compare value of feature
        (i.e. "accommodates" or "bathrooms") in across DataFrame Series
        """
        return obs2.apply(lambda x: self.calc_euclidean_dist_using_scipy(x, obs1))

    def randomise_dataframe_rows(self, df):
        """ Randomise ordering of DataFrame.
        Return a NumPy array of shuffled index values using `np.random.permutation`
        Return a new Dataframe containing the shuffled order using `loc[]`
        `seed(1)` reproduces random same results when share and run same code by others
        """
        np.random.seed(1)
        return df.loc[np.random.permutation(len(df))]
        # Alternative Approach:
        # shuffled_index = np.random.permutation(df.index)
        # return df.reindex(shuffled_index)

    def sort_dataframe_by_feature(self, df, feature):
        """ Sort DataFrame by feature (default ascending).
        Sort the DataFrame by "distance" column so there will be random order across the
        rows at the top of the list (having same lowest distance).
        """
        return df.sort_values(feature)

    def clean_price_format(self, df_price_column):
        """ Clean "price" column removing `$` `,` and `?` chars. Convert column from text to float. """
        def replace_bad_chars(row):
            row = str(row).replace(",", "")
            row = str(row).replace("$", "")
            row = str(row).replace("$", "")
            row = float(row) # .astype('float')
            return row
        return df_price_column.apply(lambda row: replace_bad_chars(row))

    def get_nearest_neighbors(self, df, model_feature_name):
        """ Filter range of nearest neighbors to select of recommended prices (TARGET_COLUMN) to charge per night for a rental listing based
        on average price of other listings based on the model feature being trained
        (i.e. "accommodates" or "bathrooms").
        """
        k_nearest_neighbors = self.prediction_config.HYPERPARAMETER_FIXED
        print("Predicted Target Column (i.e. 'price') (Avg of Nearest): %.2f (with model feature %r Avg. : %r) " % (df.iloc[0:k_nearest_neighbors][self.target_column].mean(), model_feature_name, df.iloc[0:k_nearest_neighbors][model_feature_name].mean()) )
        return df.iloc[0:k_nearest_neighbors][self.target_column].mean()

    def calc_mean_absolute_error(self, df, model_feature_name):
        """ MAE = ( |(actual1 - predicted1)| + ... + |(actualn - predictedn)| ) / n """
        column_name_predicted_target = "predicted_" + self.target_column + "_" + model_feature_name
        return df.apply(lambda x: np.absolute(x[self.target_column] - x[column_name_predicted_target]), axis=1).mean()

    def calc_mean_squared_error(self, df, model_feature_name):
        """ MSE = ( (actual1 - predicted1)^2 + ... + (actualn - predictedn)^2 ) / n """
        column_name_predicted_target = "predicted_" + self.target_column + "_" + model_feature_name
        return df.apply(lambda x: (x[self.target_column] - x[column_name_predicted_target])**2, axis=1).mean()

    def calc_root_mean_squared_error(self, df, model_feature_name):
        """ RMSE = sqrt(MSE) """
        return np.sqrt( self.calc_mean_squared_error(df, model_feature_name) )

    def generate_combinations_of_features(self, training_column_names):
        """ Generate all combinations of features without repetition

        Reduce amount of combinations by applying minimum length of MIN_FEATURES_COMBO_LEN
        """
        features = training_column_names
        loop_count = len(features)
        combos_above_min_len = list()

        def flatten_combo(combos):
            return sum(combos, [])

        if self.prediction_config.MIN_FEATURES_COMBO_LEN <= loop_count:
            i = self.prediction_config.MIN_FEATURES_COMBO_LEN
            while i >= 1 and i <= loop_count:
                combos_above_min_len.append(list(itertools.combinations(features, i)))
                i += 1
            return flatten_combo(combos_above_min_len)
        print("Error: Minimum length of any features combos cannot exceed length of all features")
        return []

    def plot(self, training_model_feature_name, testing_part):
        """ Plot """
        testing_part.pivot_table(index=training_model_feature_name, values=self.target_column).plot()
        plt.show()

    def scatter_plot_hyperparams(self, hyperparam_range, error_values):
        """ Scatter Plot hyperparameters range of 'k' versus error calculation values """

        # Plot
        colours = range(20)
        circle_size = 200
        cmap = plt.cm.viridis # plt.cm.get_cmap('jet')
        fig, ax = plt.subplots(figsize=(8, 4))
        cs = ax.scatter(x=hyperparam_range,
                    y=error_values,
                    s=circle_size,
                    c=colours,
                    marker='o',
                    cmap=cmap,
                    vmin=0.,
                    vmax=2.)
        plt.xlabel('Hyperparameter k', fontsize=18)
        plt.ylabel('MSE', fontsize=16)

        # Prettify
        ax.axis("tight")
        fig.colorbar(cs)

        plt.show()

    def plot_hyperparams(self, feature_combos_lowest_rmse_for_hyperparams):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for feature_key, dict_value in feature_combos_lowest_rmse_for_hyperparams.items():
            k = dict_value["k"]
            min_rmse = dict_value["min_rmse"]
            ax.plot(k, min_rmse, '+', label = feature_key, mew=10, ms=15)

        ax.legend(loc=0, prop={'size':6})
        ax.grid()
        ax.set_xlabel("Hyperparam k")
        yLabel = "RMSE of Features Combination using " + str(self.prediction_config.K_FOLDS) + " K-Folds for Cross Validation"
        ax.set_ylabel(yLabel)
        ax.set_ylim(-100, 20000) # RMSE
        # ax.set_ylim(-50, 20000) # MSE
        plt.show()