import pandas as pd
from sklearn.cluster import KMeans
import math
import numpy as np

class ClusteringModelKMeansExternal:
    """ External (Scikit-Learn) Machine Learning Model for K-Means Clustering """
    def __init__(self, prediction_config, prediction_data, prediction_utils):
        self.prediction_config = prediction_config
        self.prediction_data = prediction_data
        self.prediction_utils = prediction_utils
        self.model_type = "kmeans"

    def example_plot_outliers(self, df, affiliation_column, labels, clustered_row_distances):
        # Find out why some switched using Subsetting the DataFrame using Pandas to only select rows in DataFrame
        # from unique row values that switched (i.e. outliers) from their Affiliation column (i.e. 'party') and
        # matching specific Label (i.e. cluster id) of those that switched affiliation
        democratic_outliers = df[(labels == 1) & (df[affiliation_column] == "D")]
        print("Democratic Outliers: %r" % (democratic_outliers))

        import matplotlib.pyplot as plt

        # Plot the position of each row as
        #  - `x` (1st column of `clustered_row_distances` Numpy array `clustered_row_distances[:,0]`) and
        #  - `y` (2nd column of `clustered_row_distances`)
        #  - `c` as the `labels` (clusters)
        #
        # Where coordinates (since distances are relative to the Cluster Centers),
        # based on computed `clustered_row_distances` array that showed the distance from each row
        # to center of each Cluster. Shade each point according to party affiliation (so can quickly inspect
        # the layout of the different rows and see who are the outliers that switched affiliation)
        plt.scatter(x=clustered_row_distances[:,0], y=clustered_row_distances[:,1], c=labels)
        plt.show()

    def process_clustering(self):
        print("K-Means Clustering in progress...")

        if not "affiliation_column" in self.prediction_config.DATASET_LOCATION[self.prediction_config.DATASET_CHOICE]:
            return

        # Explore loaded data
        df = self.prediction_data
        target_column = self.prediction_config.DATASET_LOCATION[self.prediction_config.DATASET_CHOICE]["target_column"]
        affiliation_column = self.prediction_config.DATASET_LOCATION[self.prediction_config.DATASET_CHOICE]["affiliation_column"]

        centroids_quantity = self.prediction_config.CENTROIDS_QUANTITY
        # Initialise K-Means Clustering Model using specified quantity of clusters (centroids)
        # for training the model using the whole dataset.
        kmeans_model = KMeans(n_clusters=centroids_quantity, random_state=1)

        df_numeric = df.select_dtypes(include=['int', 'int64', 'float64', 'floating'], exclude=['O'])
        print("Excluding non-numeric columns from K-Means Clustering: ", df.select_dtypes(include=['O']).columns.tolist())

        # Fit the K-Means Model to the DataFrame to calculate the Euclidean Distance of each row
        # to each cluster (centroid) and return a Numpy array with n_columns. Each column represents a
        # cluster (centroid) and indicates how far each rows is from the nearest cluster (centroid)
        # Important Note: Pass only numeric dataframe columns
        clustered_row_distances = kmeans_model.fit_transform(df_numeric)

        # Explore clusters to by computing cross-tabulation of the quantity of rows in each clustered_row_distance column
        # and the checking how they corresponded to unique row values of Affiliation column (i.e. 'party')
        labels = kmeans_model.labels_
        print("Cross Tabulation between Clustered Labels and Affiliation i.e. 'party' column: \n%r" % (pd.crosstab(labels, df[affiliation_column])))

        # self.example_plot_outliers(df, affiliation_column, labels, clustered_row_distances)

        # Generate new DataFrame column to be used as Target Column for Prediction Algorithms
        # (i.e. to detect which roll call votes were most likely to cause extremism such
        # that Senators would not vote along their own party lines)
        extremism = (clustered_row_distances ** 3).sum(axis=1)
        df["extremism"] = extremism
        df.sort_values("extremism", inplace=True, ascending=False)
        print("Top 10 observations ranked in order of 'extremism': %r" % (df.head(10)))
        self.prediction_data.df_listings = df

def run(prediction_config, prediction_data, prediction_utils):
    """
    Scikit-Learn Workflow depending on config chosen
    """
    clustering_model_kmeans_external = ClusteringModelKMeansExternal(prediction_config, prediction_data, prediction_utils)
    clustering_model_kmeans_external.process_clustering()

