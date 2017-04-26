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

# Load downloaded copy of dataset (.csv format) into Dataframe (DF)
# `other_listings` of Pandas. Otherwise load remote (slower)
try:
    dataset_file = Path(data_set_local)
    if dataset_file.is_file():
        other_listings = pd.read_csv(data_set_local, nrows=1)
    else:
        def exists(path):
            r = requests.head(path)
            return r.status_code == requests.codes.ok
        if exists(data_set_remote):
            other_listings = pd.read_csv(data_set_remote, nrows=1)
except Exception as e:
    print(e.errno)

# Show first column (i.e. `0`) of sliced first row (i.e. `0:1`)
print(other_listings.iloc[0:1, 0])

# Access first row of DataFrame as Series from given data set.
# Access specific "feature" (column) value
other_listing_feature_accommodates_first = other_listings.iloc[0:1]["accommodates"][0]
my_listing_feature_accommodates = 3

# Euclidean Distance equation
first_distance = math.sqrt(abs(other_listing_feature_accommodates_first - my_listing_feature_accommodates)**2)
print(first_distance)

# Show first two rows with "accommodates" column
first_two_listings = other_listings.iloc[0:2]["accommodates"]
print(first_two_listings)