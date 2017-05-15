EVENT = {
    "model_workflow_config": {
        "model_workflow_for_knn_algorithm": "scikit",
        "model_workflow_for_linear_regression_algorithm_toggle": True,
        # Warning: Only change to True when Target column only contains Categorical int64 values (not float64)
        # otherwise you will get a 'continous' error since it will perform Logistic Regression.
        # i.e. For car-listing-fuel Dataset, the Target Column may be "origin", but NOT "mpg"
        "model_workflow_for_logistic_regression_algorithm_toggle": False,
    },
    "training_config": {
        "min_training_features": 2
    },
    "cleansing_config": {
        "min_percentage_incomplete_observations_to_remove_column": 0.2,
        "max_percentage_incomplete_observations_to_retain_column_and_remove_incomplete_slice": 0.02
    },
    "hyperparameter_optimisation_config": {
        "hyperparameter_optimisation_toggle": True,
        "hyperparameter_range": 20,
        "hyperparameter_quantity_fixed": 5
    },
    "k_means_clustering_config": {
        "k_means_clustering_toggle": True,
        "centroids_quantity": 2
    },
    "k_fold_cross_validation_config": {
        "k_fold_cross_validation_toggle": True,
        "k_folds_quantity": 10,
        "k_folds_workflow": "scikit"
    },
    "plot_config": {
        "plot_individual_train_features_vs_target_toggle": False,
        "plot_linear_relationship_prediction_vs_actual_for_train_features_vs_target_toggle": True,
        "plot_logistic_relationship_prediction_vs_actual_for_train_features_vs_target_toggle": True,
        "plot_logistic_roc": True,
        "plot_hyperparameter_optimisation": True
    },
    "dataset_selected": "senators-vote", # rental-property-listings, car-listings, car-listings-fuel, university-admissions, senators-vote
    "dataset_config": {
        "rental-property-listings": {
            "local": "data/listings.csv",
            "remote": "http://data.insideairbnb.com/united-states/dc/washington-dc/2015-10-03/data/listings.csv",
            "format": "csv-comma-separated",
            "labels": "",
            "exclude_columns": {
                "non_numeric": ["host_response_time", "host_response_rate", "host_acceptance_rate",
                                "host_is_superhost", "host_listings_count", "host_verifications",
                                "host_has_profile_pic", "host_identity_verified", "property_type",
                                "room_type", "bed_type", "amenities", "calendar_updated", "has_availability",
                                "requires_license", "license", "instant_bookable", "cancellation_policy",
                                "require_guest_profile_picture", "require_guest_phone_verification"],
                "non_ordinal": ["latitude", "longitude", "zipcode"],
                # Note: Dataset uses different spelling for `neighborhood_overview` (than other neighborhood columns)
                "out_of_scope": ["id", "listing_url", "scrape_id", "last_scraped", "name", "summary", "space",
                                 "description", "experiences_offered", "neighborhood_overview", "notes", "transit",
                                 "thumbnail_url", "medium_url", "picture_url", "xl_picture_url",
                                 "host_id", "host_url", "host_name", "host_since", "host_location",
                                 "host_about", "host_thumbnail_url",
                                 "host_picture_url", "host_neighbourhood", "street", "neighbourhood",
                                 "neighbourhood_cleansed", "neighbourhood_group_cleansed", "city",
                                 "state", "market", "smart_location", "country_code",
                                 "country", "is_location_exact", "calendar_last_scraped", "first_review",
                                 "last_review", "jurisdiction_names"]
            },
            # Maxiumum of 4 otherwise computer may freeze during K-Fold combinations!
            "training_columns": ["accommodates", "bedrooms", "bathrooms", "number_of_reviews"],
            "target_column": "price",
            "cleanse_columns_price_format": ["price", "weekly_price", "monthly_price", "security_deposit",
                                             "cleaning_fee", "extra_people"],
            "convert_columns_words_to_digits": []
        },
        "car-listings": {
            "local": "data/imports-85.data",
            "remote": "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
            "format": "csv-comma-separated",
            "labels": "symboling,normalized-losses,make,fuel-type,aspiration,num-of-doors,body-style,drive-wheels,engine-location,wheel-base,length,width,height,curb-weight,engine-type,num-of-cylinders,engine-size,fuel-system,bore,stroke,compression-ratio,horsepower,peak-rpm,city-mpg,highway-mpg,price",
            "exclude_columns": {
                "non_numeric": ["make", "fuel-type", "aspiration", "body-style", "drive-wheels", "engine-location", "engine-type", "fuel-system"],
                "non_ordinal": [],
                "out_of_scope": ["symboling", "normalized-losses"]
            },
            "training_columns": ["num-of-doors", "curb-weight", "horsepower", "city-mpg", "highway-mpg"],
            "target_column": "price",
            "cleanse_columns_price_format": ["price"],
            "convert_columns_words_to_digits": ["num-of-doors", "num-of-cylinders"]
        },
        # https://archive.ics.uci.edu/ml/datasets/Auto+MPG
        "car-listings-fuel": {
            "local": "data/auto-mpg.data",
            "remote": "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
            "format": "csv-whitespace-separated",
            # model-year (year car model released), origin (0 Nth America, 1 Europe, 2 Asia)
            "labels": "mpg,cylinders,displacement,horsepower,weight,acceleration,model-year,origin,car-name",
            "exclude_columns": {
                "non_numeric": ["car-name"],
                "non_ordinal": [],
                "out_of_scope": []
            },
            # i.e. ["weight", "acceleration", "displacement"]
            "training_columns": [],
            "target_column": "mpg",
            "cleanse_columns_price_format": [],
            "convert_columns_words_to_digits": []
        },
        "university-admissions": {
            "local": "data/applicants.csv",
            "remote": "https://dsserver-prod-resources-1.s3.amazonaws.com/20/admissions.csv",
            "format": "csv-comma-separated",
            "labels": "",
            "exclude_columns": {
                "non_numeric": [],
                "non_ordinal": [],
                "out_of_scope": []
            },
            # i.e. ["gpa", "gre"]
            "training_columns": [],
            "target_column": "admit",
            "cleanse_columns_price_format": [],
            "convert_columns_words_to_digits": []
        },
        "senators-vote": {
            "local": "data/114_congress.csv",
            "remote": "",
            "format": "csv-comma-separated",
            "labels": "",
            "exclude_columns": {
                "non_numeric": [],
                "non_ordinal": [],
                "out_of_scope": []
            },
            "training_columns": ["vote-bill1", "vote-bill4", "vote-bill5", "vote-bill6", "vote-bill7", "vote-bill8"],
            "target_column": "extremism",
            "affiliation_column": "party",
            "cleanse_columns_price_format": [],
            "convert_columns_words_to_digits": []
        }
    }
}