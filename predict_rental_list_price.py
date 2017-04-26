import pandas as pd
import numpy as np
import math
from pathlib import Path
import requests
import matplotlib.pyplot as plt

"""
Given you have a rental listing that accommodates up to 3 rooms.
And given a data set that contains features (column attributes) of other rental listings.
Find the optimum rental listing price using similarity metrics
"""

# Declare local and remote locations of data set.
dataset_local = "data/listing.csv"
dataset_remote = "http://data.insideairbnb.com/united-states/dc/washington-dc/2015-10-03/data/listings.csv"

def load_dataset(num_rows):
    """
    Load downloaded copy of dataset (.csv format) into Dataframe (DF)
    `other_listings` of Pandas. Otherwise load remote (slower)
    """
    try:
        dataset_file = Path(dataset_local)
        if dataset_file.is_file():
            return pd.read_csv(dataset_local, nrows=num_rows)
        else:
            def exists(path):
                r = requests.head(path)
                return r.status_code == requests.codes.ok
            if exists(dataset_remote):
                return pd.read_csv(dataset_remote, nrows=num_rows)
        return None
    except Exception as e:
        print(e.errno)

def calc_euclidean_dist(val1, val2):
    """ Euclidean Distance equation to compare values of different data sets """
    return int(math.sqrt(abs(val1 - val2)**2))

def compare_observations(obs1, obs2):
    """
    Similarity Metric that compares two observations' data set features (columns)
    and returns distance (difference)
    """
    return obs2.apply(lambda x: calc_euclidean_dist(x, obs1))

def randomise_dataframe_rows(dataframe):
    """
    Randomise the ordering of the other data set.
    Return a NumPy array of shuffled index values using `np.random.permutation`
    Return a new Dataframe containing the shuffled order using `loc[]`

    Ref: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html
    """
    # Reproduce random results when share and run same code by others
    np.random.seed(1)
    return dataframe.loc[np.random.permutation(len(dataframe))]

def sort_dataframe_by_feature(dataframe, feature):
    """ Sort dataframe by feature (default ascending) """
    return dataframe.sort_values(feature)

def clean_price(dataframe):
    """
    Clean "price" column to remove `$` and `,` characters and
    convert column from text to numeric float type.
    """
    def replace_bad_chars(row):
        row = row.replace(",", "")
        row = row.replace("$", "")
        row = float(row)
        return row

    dataframe["price"] = dataframe["price"].apply(lambda row: replace_bad_chars(row))

    return dataframe

def get_price_prediction(my_listing_accommodates_first, other_listings_all_accommodates):

    # Compare value of feature "accommodates" in DataFrame Series from other data set with my data set value.
    # Assign distance values to new "distance" column of Data Frame Series object.
    other_listings_all["distance"] = compare_observations(my_listing_accommodates_first, other_listings_all_accommodates)

    # Use the Panda Series method value_counts to display unique value counts for each "distance" column.
    # Ascending order by "distance".
    # print(other_listings_all["distance"].value_counts()) # .index.tolist()

    # Check the value count for "distance" value of 0. Its value 461 is amount of
    # other rental listings that also accommodate 3 people.
    # Avoid bias (toward just the sort order by "distance" column of the data set) when choosing
    # the five (5) "nearest neighbors" (all will have distance 0 since only want 5 and there
    # are 461 indexes with that distance).

    # Show all listing indexes in the other data set that have a distance of 0 from my data set
    # (i.e. also accommodating 3 people) for the feature "accommodates"
    # print(other_listings_all[other_listings_all["distance"] == 0]["accommodates"])

    # Randomise the ordering of the other data set first, and only then:
    # Sort the DataFrame by "distance" column
    # so there will be random order across the first 461 rows (having lowest distance)
    other_listings_all_randomised = randomise_dataframe_rows(other_listings_all)
    other_listings_all_sorted = sort_dataframe_by_feature(other_listings_all_randomised, "distance")

    # Show first 10 values in "price" column
    # print(other_listings_all_sorted.iloc[0:10]["price"])

    # Convert new Series object containing cleaned values to float datatype.
    # Assign back to "price" column in data set.
    other_listings_all_cleaned = clean_price(other_listings_all_sorted)
    # print(other_listings_all_cleaned)

    other_listings_all_cleaned.pivot_table(index='accommodates', values='price').plot()
    plt.show()

    # Select "nearest neighbors" (first 5 values in "price" column)
    # Assign to `mean_price` the mean of the `price` column
    mean_price = other_listings_all_cleaned.iloc[0:5]["price"].mean()

    # Show `mean_price`
    print(mean_price)

    return mean_price

# Loads other data set as DataFrame with all rows
other_listings_all = load_dataset(None)

# Convert my data set from dict to DataFrame as emulation
my_dataset = { "accommodates": [3] }
my_listing = pd.DataFrame.from_dict(my_dataset)

# Fetch value in first row and feature "accommodates" from my dataset.
# Fetch all rows of feature "accommodates" from other dataset.
my_listing_accommodates_first = my_listing.iloc[0:1]["accommodates"][0]
other_listings_all_accommodates = other_listings_all["accommodates"]

# Find recommended price to charge per night for my rental listing based
# on average price of other listings that accommodate 3 people.
price_recommended = get_price_prediction(my_listing_accommodates_first, other_listings_all_accommodates)
