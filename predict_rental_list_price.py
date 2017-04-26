import pandas as pd
import math
from pathlib import Path
import requests

"""
Given you have a rental listing that accommodates up to 3 rooms.
And given a data set that contains features (column attributes) of other rental listings.
Find the optimum rental listing price using similarity metrics
"""

# Declare local and remote locations of data set.
data_set_local = "data/listing.csv"
data_set_remote = "http://data.insideairbnb.com/united-states/dc/washington-dc/2015-10-03/data/listings.csv"

def load_data_set(num_rows):
    """
    Load downloaded copy of dataset (.csv format) into Dataframe (DF)
    `other_listings` of Pandas. Otherwise load remote (slower)
    """
    try:
        dataset_file = Path(data_set_local)
        if dataset_file.is_file():
            return pd.read_csv(data_set_local, nrows=num_rows)
        else:
            def exists(path):
                r = requests.head(path)
                return r.status_code == requests.codes.ok
            if exists(data_set_remote):
                return pd.read_csv(data_set_remote, nrows=num_rows)
        return None
    except Exception as e:
        print(e.errno)

def calc_euclidean_dist(val1, val2):
    """
    Euclidean Distance equation to compare values of different data sets
    """
    return int(math.sqrt(abs(val1 - val2)**2))

def compare_observations(obs1, obs2):
    """
    Similarity Metric that compares two observations' data set features (columns)
    and returns distance (difference)
    """
    return obs2.apply(lambda x: calc_euclidean_dist(x, obs1))

# Loads other data set as DataFrame with all rows
other_listings_all = load_data_set(None)

# Convert my data set from dict to DataFrame as emulation
my_data_set = { "accommodates": [3] }
my_listing = pd.DataFrame.from_dict(my_data_set)

# Fetch from my data set value of feature (column) "accommodates" in first row.
# Fetch from my data set value of feature (column) "accommodates" all rows.
my_listing_accommodates_first = my_listing.iloc[0:1]["accommodates"][0]
other_listings_all_accommodates = other_listings_all["accommodates"]

# Compare value of feature "accommodates" in DataFrame Series from other data set with my data set value.
# Assign distance values to new "distance" column of Data Frame Series object.
other_listings_all["distance"] = compare_observations(my_listing_accommodates_first, other_listings_all_accommodates)

# Use the Panda Series method value_counts to display unique value counts for "distance" column
print(other_listings_all["distance"].value_counts()) # .index.tolist()
