import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

class PredictionUtils():
    """ Utility functions """

    def __init__(self, prediction_config):
        self.prediction_config = prediction_config
        self.target_column = self.prediction_config.DATASET_LOCATION[self.prediction_config.DATASET_CHOICE]["target_column"]

    def generate_model(self, regressor, qty_neighbors, algorithm, distance_type):
        """ Regressor Model Generation"""
        if regressor == "knn":
            return KNeighborsRegressor(n_neighbors=qty_neighbors, algorithm=algorithm, p=distance_type)
        elif regressor == "linear":
            return LinearRegression(fit_intercept=True) # copy_X=True, n_jobs=1, normalize=False
        elif regressor == "logistic":
            return LogisticRegression(class_weight='balanced')

    def k_fold_cross_validation(self, regressor, df, feature_combos):
        """ K-Fold Cross Validation for any given Regressor """
        feature_combos_rmse_for_hyperparams = dict()
        fold_ids = list(range(1, self.prediction_config.K_FOLDS + 1))

        for idx1, feature_combo in enumerate(feature_combos):
            feature_combo_key = '__'.join(feature_combo)
            feature_combos_rmse_for_hyperparams[feature_combo_key] = list()
            for idx2, qty_neighbors in enumerate(self.prediction_config.HYPERPARAMETER_RANGE):
                fold_rmses = []

                def cross_validation_manual():
                    """ Manual KFold Cross Validation using 'fold' column """
                    for fold in fold_ids:
                        # Train
                        model = self.generate_model(regressor, qty_neighbors, 'brute', 2)
                        train_part = df[df["fold"] != fold]
                        test_part = df[df["fold"] == fold]
                        X = train_part[list(feature_combo)]
                        y = train_part[self.target_column]
                        model.fit(X, y)
                        # Predict
                        predictions = model.predict(test_part[list(feature_combo)])
                        # test_part["predicted_price"] = predictions
                        mse = mean_squared_error(test_part[self.target_column], predictions)
                        rmse = mse**(1/2)
                        fold_rmses.append(rmse)
                    return np.mean(fold_rmses)

                def cross_validation_with_builtin():
                    """ Scikit-Learn Built-in KFold class to generate KFolds and run Cross Validation
                    with training using Scikit-Learn Built-in `cross_val_score` function"""
                    kf = KFold(n_splits=self.prediction_config.K_FOLDS, shuffle=True, random_state=8)
                    model = self.generate_model(regressor, qty_neighbors, 'brute', 2)

                    if (self.prediction_config.K_FOLDS and len(df)) and self.prediction_config.K_FOLDS <= len(df):
                        # MSEs for each Fold
                        mses = cross_val_score(model, df[list(feature_combo)], df[self.target_column], scoring="neg_mean_squared_error", cv=kf, verbose=0)
                        fold_rmses = [np.sqrt(np.absolute(mse)) for mse in mses]
                        return np.mean(fold_rmses)
                    else:
                        return None

                if self.prediction_config.K_FOLDS_BUILTIN == "manual":
                    avg_rmse = cross_validation_manual()
                else:
                    avg_rmse = cross_validation_with_builtin()
                # print("Fold RMSEs %r: " % (fold_rmses))
                print("K-Fold Cross Validation found Average RMSE using Regressor %r for Feature Combo %r with Hyperparam k of %r using %r K-Folds: %r" % (regressor, feature_combo_key, qty_neighbors, self.prediction_config.K_FOLDS, avg_rmse))
                feature_combos_rmse_for_hyperparams[feature_combo_key].append(avg_rmse)
        return feature_combos_rmse_for_hyperparams

    def hyperparameter_k_optimisation(self, feature_combos_rmse_for_hyperparams, model_type, pre_optimisation_results):
        """ Hyperparameter k Optimisation """
        feature_combos_lowest_rmse_for_hyperparams = dict()

        for key, value in feature_combos_rmse_for_hyperparams.items():
            # Initiate first element to key for lowest RMSE. If find an even lower RMSE at subsequent index it will be replaced
            feature_combos_lowest_rmse_for_hyperparams[key] = dict()
            feature_combos_lowest_rmse_for_hyperparams[key]["min_rmse"] = feature_combos_rmse_for_hyperparams[key][0]
            feature_combos_lowest_rmse_for_hyperparams[key]["k"] = 1
            for k, rmse in enumerate(feature_combos_rmse_for_hyperparams[key]):
                if rmse and feature_combos_lowest_rmse_for_hyperparams[key]["min_rmse"]:
                    if rmse < feature_combos_lowest_rmse_for_hyperparams[key]["min_rmse"]:
                        feature_combos_lowest_rmse_for_hyperparams[key]["min_rmse"] = rmse
                        feature_combos_lowest_rmse_for_hyperparams[key]["k"] = k + 1

        # Find best combination of hyperparameter k and features

        # Initiate element with lowest RMSE as first element unless find a lower element at subsequent index
        name_of_first_key = list(feature_combos_lowest_rmse_for_hyperparams.keys())[0]
        feature_combo_name_with_lowest_rmse = name_of_first_key
        lowest_rmse = feature_combos_lowest_rmse_for_hyperparams[name_of_first_key]["min_rmse"]
        highest_rmse = feature_combos_lowest_rmse_for_hyperparams[name_of_first_key]["min_rmse"]
        k_value_of_lowest_rmse = feature_combos_lowest_rmse_for_hyperparams[name_of_first_key]["k"]

        for feature_key, dict_value in feature_combos_lowest_rmse_for_hyperparams.items():
            if highest_rmse and (dict_value["min_rmse"] >= highest_rmse):
                highest_rmse = dict_value["min_rmse"]
            if lowest_rmse and (dict_value["min_rmse"] < lowest_rmse):
                feature_combo_name_with_lowest_rmse = feature_key
                lowest_rmse = dict_value["min_rmse"]
                k_value_of_lowest_rmse = dict_value["k"]
        print("Feature combo %r has lowest RMSE of %r with 'k' of %r (optimum) using %r K-Folds for (Cross Validation was %r)" % (feature_combo_name_with_lowest_rmse, lowest_rmse, k_value_of_lowest_rmse, self.prediction_config.K_FOLDS, self.prediction_config.K_FOLD_CROSS_VALIDATION) )

        if not self.prediction_config.SUPPRESS_ALL_PLOTS and self.prediction_config.PLOT_HYPERPARAMETER_OPTIMISATION == True:
            self.plot_hyperparams(feature_combos_lowest_rmse_for_hyperparams, feature_combo_name_with_lowest_rmse, k_value_of_lowest_rmse, lowest_rmse, highest_rmse, model_type, pre_optimisation_results)
        return {
            "feature_combo_name_with_lowest_rmse": feature_combo_name_with_lowest_rmse,
            "lowest_rmse": lowest_rmse,
            "k_value_of_lowest_rmse": k_value_of_lowest_rmse
        }

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

    def calc_sensitivity(self, count_true_positives, count_false_negatives):
        """ Calculate Sensitivity aka True Positive Rate (TPR) to find out
        how effective model is at identifying positive outcomes
        (i.e. of all the Predicted observations that should have been admitted as 1 to matched the
        Actual observation, what fraction did the Binary Classification Model correctly predict as 1)

        If Sensitivity is 10% then only 1 in 10 observations that should have had a
        Predicted column row value of 1 (to match the actual Target column row) where actually given a
        Predicted column row value of 1
        """
        return count_true_positives / (count_true_positives + count_false_negatives)

    def calc_specificity(self, count_true_negatives, count_false_positives):
        """ Calculate Specificity aka True Negative Rate (TNR) to find out
        how effective model is at identifying negative outcomes
        """
        return count_true_negatives / (count_true_negatives + count_false_positives)

    def calc_binary_classification(self, predicted_target_values, actual_target_values):
        true_positive_filter = (predicted_target_values == 1) & (actual_target_values == 1)
        count_true_positives = len(true_positive_filter)
        true_negative_filter = (predicted_target_values == 0) & (actual_target_values == 0)
        count_true_negatives = len(true_negative_filter)
        false_positive_filter = (predicted_target_values == 1) & (actual_target_values == 0)
        count_false_positives = len(false_positive_filter)
        false_negative_filter = (predicted_target_values == 0) & (actual_target_values == 1)
        count_false_negatives = len(false_negative_filter)
        return {
            "sensitivity": self.calc_sensitivity(count_true_positives, count_false_negatives),
            "specificity": self.calc_specificity(count_true_negatives, count_false_positives)
        }

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
        if self.prediction_config.SUPPRESS_ALL_PLOTS or not training_model_feature_name or isinstance(testing_part, type(None)):
            return
        testing_part.pivot_table(index=training_model_feature_name, values=self.target_column).plot()
        plt.show(block=False) # Avoid plots conflicting
        plt.show()

    def plot_corr(self, df, size=10):
        """
        Function plots a graphical correlation matrix for each pair of columns in the dataframe.

        Input:
            df: pandas DataFrame
            size: vertical and horizontal size of the plot
        """
        if self.prediction_config.SUPPRESS_ALL_PLOTS or not self.prediction_config.PLOT_CORRELATION_BETWEEN_TARGET_COLUMN_AND_OTHERS:
            return

        corr = df.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns, fontsize=7, rotation=75)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
        plt.show(block=False) # Avoid plots conflicting
        plt.show()

    def scatter_plot_hyperparams(self, hyperparam_range, error_values):
        """ Scatter Plot hyperparameters range of 'k' versus error calculation values """
        if self.prediction_config.SUPPRESS_ALL_PLOTS or not hyperparam_range or not error_values:
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

        plt.show(block=False) # Avoid plots conflicting
        plt.show()

    def plot_hyperparams(self, feature_combos_lowest_rmse_for_hyperparams, feature_combo_name_with_lowest_rmse, k_value_of_lowest_rmse, lowest_rmse, highest_rmse, model_type, pre_optimisation_results):

        if self.prediction_config.SUPPRESS_ALL_PLOTS or not feature_combos_lowest_rmse_for_hyperparams or not feature_combo_name_with_lowest_rmse or not lowest_rmse or not highest_rmse:
            return

        count_feature_combos = len(feature_combos_lowest_rmse_for_hyperparams.items())

        fig = plt.figure()
        title = 'Hyperparameter k Optimisation Results for regression: ' + str(model_type)
        fig.suptitle(title, fontsize=14, fontweight='bold')
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
        ax.set_xlabel("Hyperparam k", fontsize=12)
        yLabel = "RMSE of Features Combinations"
        ax.set_ylabel(yLabel, fontsize=12)

        target_column = "Target Feature: " + self.prediction_config.DATASET_LOCATION[self.prediction_config.DATASET_CHOICE]["target_column"]
        lowest_rmse_text = "Lowest RMSE: " + str(round(lowest_rmse, 2))
        lowest_rmse_feature_combination = "Best Feature Combo: " + feature_combo_name_with_lowest_rmse
        lowest_rmse_hyperparameter_k = "Hyperparameter k of best result: " + str(k_value_of_lowest_rmse)
        k_folds_text = "K-Folds: " + str(self.prediction_config.K_FOLDS)
        k_nearest_neighbors_range_text = "Hyperparameter k Range: 0 to " + str(self.prediction_config.HYPERPARAMETER_RANGE[-1])

        pre_optimisation_results_summary = ""

        if pre_optimisation_results:
            _model_type = ""
            _rmse = ""
            _accuracy = "N/A" # Default is N/A
            _sensitivity = "N/A" # Default is N/A
            _specificity = "N/A" # Default is N/A
            _auc_score = "N/A" # Default is N/A
            if "model_type" in pre_optimisation_results:
                _model_type = str(pre_optimisation_results["model_type"])

            if "rmse" in pre_optimisation_results:
                _rmse = str(pre_optimisation_results["rmse"])

            if "accuracy" in pre_optimisation_results:
                _accuracy = str(pre_optimisation_results["accuracy"])

            if "sensitivity" in pre_optimisation_results:
                _sensitivity = str(pre_optimisation_results["sensitivity"])

            if "specificity" in pre_optimisation_results:
                _specificity = str(pre_optimisation_results["specificity"])

            if "auc_score" in pre_optimisation_results:
                _auc_score = str(pre_optimisation_results["auc_score"])

            pre_optimisation_results_summary = "Pre-Optimisation - " + \
                                               "Model Type: " + _model_type + "; " + \
                                               "RMSE: " + _rmse + "; " + \
                                               "\nAccuracy: " + _accuracy + "; " + \
                                               "Sensitivity: " + _sensitivity + "; " + \
                                               "Specificity: " + _specificity + "; " + \
                                               "\nAUC Score: " + _auc_score

        results_text = target_column + "\n" + \
                       lowest_rmse_text + "\n" + \
                       lowest_rmse_feature_combination + "\n" + \
                       lowest_rmse_hyperparameter_k + "\n" + \
                       k_folds_text + "\n" + \
                       k_nearest_neighbors_range_text + "\n" + \
                       pre_optimisation_results_summary
        ax.text(0.5, 1.2, results_text, style='italic',
                bbox={'facecolor':'red', 'alpha':0.2, 'pad':5},
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=7)

        # Optimise plot to dynamically fit all values regardless of range of RMSE results
        if highest_rmse and lowest_rmse:
            contingency = highest_rmse * 0.1
            highest_rmse_with_contingency = highest_rmse + contingency
            lowest_rmse_with_contingency = lowest_rmse - contingency
            ax.set_ylim(lowest_rmse_with_contingency, highest_rmse_with_contingency) # RMSE

        # Legend squeezed on right side
        plt.legend(bbox_to_anchor=(2,1), loc="upper right", ncol=2, mode="expand", borderaxespad=0, prop={'size':5})
        # plt.subplots_adjust(top=0.7)
        plt.tight_layout(rect=[0,0,0.5,1])
        fig.subplots_adjust(top=0.7)  # subplots_adjust must be after call to tight_layout
        plt.show(block=False) # Avoid plots conflicting
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
        if self.prediction_config.SUPPRESS_ALL_PLOTS or isinstance(df, type(None)) or not training_columns or isinstance(predictions, type(None)):
            return
        training_columns_df = df.filter(training_columns, axis=1)
        count_subplots = len(training_columns_df.columns)
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle('Evaluate Linear Relationships Results', fontsize=14, fontweight='bold')
        count_columns = 2
        for index, col in enumerate(training_columns_df.columns):
            """ Subplots i.e. 2 rows x 3 columns grid at 4th subplot """
            subplot = index + 1
            ax = fig.add_subplot(count_subplots, count_columns, subplot)
            # plt.title("Training Column vs Target Column", fontsize=10)
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
        fig.subplots_adjust(top=0.85)
        plt.show(block=False) # Avoid plots conflicting
        plt.show()

    def plot_logistic_relationship_comparison(self, df, training_columns, positive_predictions_probabilities):
        """"""
        if self.prediction_config.SUPPRESS_ALL_PLOTS or isinstance(df, type(None)) or not training_columns or isinstance(positive_predictions_probabilities, type(None)):
            return
        training_columns_df = df.filter(training_columns, axis=1)
        count_subplots = len(training_columns_df.columns)
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle('Evaluate Logistic Relationships Results', fontsize=14, fontweight='bold')
        count_columns = 2
        for index, col in enumerate(training_columns_df.columns):
            """ Subplots i.e. 2 rows x 3 columns grid at 4th subplot """
            subplot = index + 1
            ax = fig.add_subplot(count_subplots, count_columns, subplot)
            x = col
            y = self.target_column
            label_actual = col + ' (actual)'
            label_predicted = y + ' (positive predicted probabilities)'
            ax.scatter(df[x], df[y], c='red', marker='o', label=label_actual)
            ax.scatter(df[x], positive_predictions_probabilities, c='blue', marker='o', label=label_predicted)
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
        fig.subplots_adjust(top=0.85)
        plt.show(block=False) # Avoid plots conflicting
        plt.show()

    def plot_receiver_operator_characteristic(self, fpr, tpr, auc_score):

        if self.prediction_config.SUPPRESS_ALL_PLOTS:
            return

        fig = plt.figure()
        fig.suptitle('ROC Curve', fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111)
        # ax.set_title('ax title')
        ax.set_xlabel('Fall-Out aka False Positive Rate (FPR)', fontsize=12)
        ax.set_ylabel('Sensitivity aka True Positive Rate (TPR)', fontsize=12)
        auc_score_text = "AUC Score: " + str(auc_score)
        ax.text(fpr.mean(), tpr.mean(), auc_score_text, style='italic',
                bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
        plt.plot(fpr, tpr)
        plt.show(block=False) # Avoid plots conflicting
        plt.show()