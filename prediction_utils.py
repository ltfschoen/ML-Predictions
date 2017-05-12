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
        if isinstance(df, type(None)):
            return None
        return (df - df.mean()) / (df.std())

    def get_percentage_missing(self, df, column):
        """ Calculates percentage of NaN values in DataFrame
        :param series: Pandas DataFrame object
        :return: float
        """
        if isinstance(df, type(None)):
            return None

        try:
            count_question_marks = 0
            for i, v in enumerate(df[column]):
                if df[column][i] == "?":
                    count_question_marks += 1
            num = df[column].isnull().sum() + count_question_marks
            den = len(df[column])
            col_percentage_missing = round(num/den, 2)
            if col_percentage_missing is not None:
                return float(round(num/den, 2))
            else:
                return 0.0
        except Exception as e:
            print("Error: Unable to get percentage missing %r: %r" % (column, e))
            return 0.0

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
        if isinstance(df, type(None)):
            return None
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
        if isinstance(df, type(None)):
            return None
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
        if isinstance(df, type(None)):
            return None
        k_nearest_neighbors = self.prediction_config.HYPERPARAMETER_FIXED
        print("Predicted Target Column (i.e. 'price') (Avg of Nearest): %.2f (with model feature %r Avg. : %r) " % (df.iloc[0:k_nearest_neighbors][self.target_column].mean(), model_feature_name, df.iloc[0:k_nearest_neighbors][model_feature_name].mean()) )
        return df.iloc[0:k_nearest_neighbors][self.target_column].mean()

    def calc_mean_absolute_error(self, df, model_feature_name):
        """ MAE = ( |(actual1 - predicted1)| + ... + |(actualn - predictedn)| ) / n """
        if isinstance(df, type(None)):
            return None
        column_name_predicted_target = "predicted_" + self.target_column + "_" + model_feature_name
        return df.apply(lambda x: np.absolute(x[self.target_column] - x[column_name_predicted_target]), axis=1).mean()

    def calc_mean_squared_error(self, df, model_feature_name):
        """ MSE = ( (actual1 - predicted1)^2 + ... + (actualn - predictedn)^2 ) / n """
        if isinstance(df, type(None)):
            return None
        column_name_predicted_target = "predicted_" + self.target_column + "_" + model_feature_name
        return df.apply(lambda x: (x[self.target_column] - x[column_name_predicted_target])**2, axis=1).mean()

    def calc_root_mean_squared_error(self, df, model_feature_name):
        """ RMSE = sqrt(MSE) """
        if isinstance(df, type(None)):
            return None
        return np.sqrt( self.calc_mean_squared_error(df, model_feature_name) )

    def generate_combinations_of_features(self, training_column_names):
        """ Generate all combinations of features without repetition

        Reduce amount of combinations by applying minimum length of MIN_FEATURES_COMBO_LEN
        """
        if not training_column_names:
            return []

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
        raise ValueError("Error: Insufficient training feature combinations to satisfy the configured minimum quantity")

    def plot(self, training_model_feature_name, testing_part):
        """ Plot """
        if not training_model_feature_name or isinstance(testing_part, type(None)):
            return
        testing_part.pivot_table(index=training_model_feature_name, values=self.target_column).plot()
        plt.show()

    def scatter_plot_hyperparams(self, hyperparam_range, error_values):
        """ Scatter Plot hyperparameters range of 'k' versus error calculation values """
        if not hyperparam_range or not error_values:
            return
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

    def plot_hyperparams(self, feature_combos_lowest_rmse_for_hyperparams, lowest_rmse, highest_rmse):

        if not feature_combos_lowest_rmse_for_hyperparams or not lowest_rmse or not highest_rmse:
            return

        count_feature_combos = len(feature_combos_lowest_rmse_for_hyperparams.items())

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Automatically select specified number of colours from existing colourmap
        number = count_feature_combos + 1

        # Select colourmap
        cmap = plt.get_cmap('gist_ncar')
        colors = [cmap(i) for i in np.linspace(0, 1, number)]

        j = 1
        for feature_key, dict_value in feature_combos_lowest_rmse_for_hyperparams.items():
            if dict_value["k"] and dict_value["min_rmse"]:
                k = dict_value["k"]
                min_rmse = dict_value["min_rmse"]
                ax.plot(k, min_rmse, '+', color=colors[j], label = feature_key, mew=5, ms=10)
                j += 1

        ax.legend(prop={'size':4})
        ax.grid()
        ax.set_xlabel("Hyperparam k")
        yLabel = "RMSE of Features Combination using " + str(self.prediction_config.K_FOLDS) + " K-Folds for Cross Validation"
        ax.set_ylabel(yLabel)

        # Optimise plot to dynamically fit all values regardless of range of RMSE results
        if highest_rmse and lowest_rmse:
            contingency = highest_rmse * 0.1
            highest_rmse_with_contingency = highest_rmse + contingency
            lowest_rmse_with_contingency = lowest_rmse - contingency
            ax.set_ylim(lowest_rmse_with_contingency, highest_rmse_with_contingency) # RMSE

        # Legend squeezed on right side
        plt.legend(bbox_to_anchor=(1.2,1), loc="upper right", mode="expand", borderaxespad=0, prop={'size':5})
        # plt.subplots_adjust(top=0.7)
        plt.tight_layout(rect=[0,0,0.5,1])
        plt.show()

    def plot_linear_relationship_comparison(self, df, training_columns, predictions):
        """ Exploratory Data Analysis to interpret model by plotting
        ACTUAL Training column against the Target column (actual fuel efficiency values) and
        PREDICTED column (predicted fuel efficiency values trained with known/ACTUAL data)
        to get visual understanding of model effectiveness

        - Purpose:
            - Machine Learning Model is "equation" representing input to output mapping by determining relationship
              between Independent Variables and the Dependent Variable
            - Linear Machine Learning Models are of the form: y = mx + b
                - where input x is transformed using m slope and b intercept parameters
                - where output is y
                - where m expected to be negative since negative linear relationship
            - Determine relationship between Target column (Dependent Variable) and Training columns (Independent Variables)
            - Determine how Training columns affect the Target column
            - Determine which Training column best correlates to Target column
        - Process of Finding "equation" (Machine Learning Model) that best "Fits" the data
            - Interpret Linear Relationship
                - Strong / Weak and Positive or Negative Linear Relationship / Gradient
        """
        if isinstance(df, type(None)) or not training_columns or isinstance(predictions, type(None)):
            return
        training_columns_df = df.filter(training_columns, axis=1)
        count_subplots = len(training_columns_df.columns)
        fig = plt.figure(figsize=(10, 10))
        count_columns = 2
        for index, col in enumerate(training_columns_df.columns):
            """ Subplots i.e. 2 rows x 3 columns grid at 4th subplot """
            subplot = index + 1
            ax = fig.add_subplot(count_subplots, count_columns, subplot)
            x = col
            y = self.target_column
            label_actual = col + ' (actual)'
            label_predicted = y + ' (predicted)'
            ax.scatter(df[x], df[y], c='red', marker='o', label=label_actual)
            ax.scatter(df[x], predictions, c='blue', marker='o', label=label_predicted)
            ax.set_xlabel(x, fontsize=12)
            ax.set_ylabel(y, fontsize=12)

            # Optimise X-axis range
            df_x_max = df[x].max()
            df_x_min = df[x].min()
            contingency = df_x_max * 0.1
            df_x_max_with_contingency = df_x_max + contingency
            df_x_min_with_contingency = df_x_min - contingency
            ax.set_xlim(df_x_min_with_contingency, df_x_max_with_contingency)

            ax.legend(bbox_to_anchor=(-0.9,-0.02), loc="best", prop={'size':5})
        fig.tight_layout(rect=[0,0,1,1])

        plt.show()