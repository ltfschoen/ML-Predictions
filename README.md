---
Machine Learning - Predictions
---

Travic CI Build Status: [![Build Status](https://api.travis-ci.org/ltfschoen/ML-Predictions.svg)](https://travis-ci.org/ltfschoen/ML-Predictions)

# Table of Contents
  * [Chapter 0 - About](#chapter-0)
  * [Chapter 1 - Setup](#chapter-1)
  * [Chapter 2 - Tests](#chapter-2)
  * [Chapter 3 - Results](#chapter-3)

## Chapter 0 - About <a id="chapter-0"></a>

* Applies **Supervised Machine Learning** techniques on past data using a **regression** process since
we want to predict a specific optimum number value for a chosen Target feature (column) of a given dataset based on
a list of chosen Training features by using a chosen Train/Test Validation Process and
optionally performing K-Folds Cross Validation (splitting into different the Train/Test partitions),
and with Hyperparameter k Optimisation (trying different quantity of nearest neighbors to find optimum) with Machine Learning Models
using different Regressors including Logistic, Linear, and K-Nearest-Neighbors (KNN) algorithms
provided by the Scikit-Learn library, which use Similarity Metrics such as the Euclidean Distance equation to perform comparisons,
and evaluate the quality of the prediction accuracy using Error Metrics such as
Median Average Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) equations,
where RMSE penalises large errors more (caused by outliers).

When Logistic Regression is used we also calculate the Accuracy, Sensitivity, Specificity, and Area Under Curve (AUC) for predictions
and plot ROC Curve prior to K-Folds Cross Validation (to help understand the generalisability and reduce overfitting of models
by trying different values of K to get a better sense of how accurate we expect the model to perform on unseen data it was not trained on)
and Hyperparameter Optimisation.

**Unsupervised Machine Learning** algorithms including **K-Means Clustering** is used in Example 5 to find **patterns** in the data and
generates a new column in the main DataFrame that is **ordered and continuous** for subsequent use by **Supervised Machine Learning** Prediction Algorithms
as the Target Column (which did not exist prior).
Prior to using K-Means Clustering, an **"affiliation_column"** property must be added within the dataset key
and assigned the column that is associated with each cluster.

Refer to Implementation Guide in `prediction_config.py` for details.

## Chapter 1 - Setup <a id="chapter-1"></a>

* Install dependencies:
    ```
    pip freeze > requirements.txt
    pip install -r requirements.txt
    ```

* Setup image rendering [backend](http://matplotlib.org/faq/usage_faq.html#what-is-a-backend) of matplotlib on macOS:
    ```
    touch ~/.matplotlib/matplotlibrc; echo 'backend: TkAgg' >> ~/.matplotlib/matplotlibrc`
    ```

* Read the Implementation Guide in `prediction_config.py`.
* Configure desired input values into EVENT object within `input_event.py` (see example configurations in examples shown below)
    * Note: Designed in accordance with [AWS Lambda Function Programming Model](http://docs.aws.amazon.com/lambda/latest/dg/python-programming-model.html)

* Run
    ```
    python3 main.py
    ```

* Note: Change from `np.random.seed(1)` to `np.random.seed(0)` to generate different instead of
same random permutations each time its run.

## Chapter 2 - Tests <a id="chapter-2"></a>

* Run all Unittests with:
    ```
    python -m unittest discover -s ./tests -p '*_test.py'
    ```

## Chapter 3 - Results <a id="chapter-3"></a>

### Summary of Results

* KNN Regression achieves better RMSE than Linear Regression but with much higher Hyperparameter k value (up to 20x more)
and with more Training columns used.
* Logistic Regression achieves better RMSE than only Linear Regression in the "university_admissions" dataset where the
Target column contains Boolean values and where there are only two possible Training Columns. KNN Regression achieves slightly
better RMSE than Logistic Regression, but with much higher Hyperparameter k value (20 instead of 1), so would take longer
to compute on large datasets.
* Error Metrics reduce in multivariate configurations where more columns
are used to Train the KNN model and when rows with NaN values are removed. Adding all columns as features
instead of just the best four features actually reduces accuracy.

### Other Information

* RMSE indicates how inaccurate we expect the predicted price value to be on average
(i.e. if RMSE is $127 its means the model inaccurately predicts the price values by $127 on average,
which means the model's usefulness must be improved by reducing the RMSE particularly when the
average value in the "price" column may be $300)

* **About the MAE to RMSE Ratio**
    * Comparing MAE to RMSE Ratio highlights if "outliers" exist that cause large but infrequent errors.
    * Expect MAE > RMSE (since RMSE takes the square root of the squared error MAE)
    * RMSE penalises when large errors more than MAE (i.e. when dataset contains outlier observations that largely deviate from others)
    * RMSE is in the denominator of the ratio such that higher RMSE results in smaller Ratio.
    * Expect a model to contain large outliers when MAE < RMSE

### Result Statistics and Screenshots

#### Example 1: Rental Property Listing Dataset

* Problem for Dataset 1:
    * Problem (abstract) - What's the optimum nightly price (prediction) to rent a living space using existing data on local listings?
    * Problem (narrow) - What's the optimum nightly price (prediction) to rent a living space using existing data of on local listings based on their attributes including price, number of bedrooms, room type?

* Setup Configuration:
    * Check that the following property values are assigned in `input_event.py`:
        ```
        "model_workflow_config": {
            "model_workflow_for_knn_algorithm": "scikit",
            "model_workflow_for_linear_regression_algorithm_toggle": True,
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
        "dataset_selected": "rental-property-listings",
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
                "training_columns": ["accommodates", "bedrooms", "bathrooms", "number_of_reviews"],
                "target_column": "price",
                "cleanse_columns_price_format": ["price", "weekly_price", "monthly_price", "security_deposit",
                                                 "cleaning_fee", "extra_people"],
                "convert_columns_words_to_digits": []
            },
            ...
        ```

* Setup used:
    * Training columns (features): "accommodates", "bedrooms", "bathrooms", "number_of_reviews"
    * Target column: "price"
    * K-Folds for Cross Validation: 10
    * Hyperparameter k Range: 0 to 20

* **Logistic Regression**

    * **WARNING** If you use the Default Target Column we are evaluating in the demo of "price", which
    contains continuous float64 values it will not perform Logistic Regression.
    Logistic Regression requires int64 values (i.e. Boolean 1's and 0's) to be in the Target Column. Currently the
    implementation does not automatically convert the Target Column values from float64 to int64, instead
    it throws a ValueError and recommends the user update the `input_event.py` file
    and change the value to `model_workflow_for_logistic_regression_algorithm_toggle": False`, alternatively
    the user could inspect the dataset and choose a different Target Column that contains int64 values.
    Logistic Regression with a Target Column containing Boolean values is demonstrated in "university_admissions"
    dataset example.

* **Linear Regression**

    * Results:
        * **Best Feature Combo**: "accommodates", "bedrooms", "bathrooms", "number_of_reviews"
        * **Best RMSE**:
            * 109.43
        * **Pre-Optimisation Best RMSE**:
            * 111.38
        * **Hyperparameter k of best**:
            * 1

    * Screenshots:

        ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/rental_property_listing/linear_regression.png)

* **KNN Regression**

    * Key Results:
        * **Best Feature Combo**: "accommodates", "bedrooms", "bathrooms", "number_of_reviews"
        * **Best RMSE**:
            * 108.6
        * **Hyperparameter k of best**:
            * 19
        * Note: Using MSE, the best result was MSE 12718.88 with k of 17 using 10 K-Folds

    * Screenshots:

        ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/rental_property_listing/knn_regression.png)

    * Prior Results

        * **Univariate (one column) using manual KNN model with NaNs removed, with k=5 nearby neighbors**
            * Comparison of Results (of "price" vs "predicted_price"):
                * Model Trained #1 "accommodates" column:
                    * MAE: 61.42
                    * MSE: 15117.62
                    * RMSE: 122.95
                    * MAE to RMSE Ratio: 0.50:1
                * Model Trained #2 "bedrooms" column:
                    * MAE: 52.82
                    * MSE: 13663.12
                    * RMSE: 116.89
                        * Note: Expect model to be off by $116 on average for predicted price values
                    * MAE to RMSE Ratio: 0.45:1
                * Key Changes:
                    * Removal of columns with >20% of its rows being NaN
                    * Columns with <1% of NaNs having the shared row/observation were removed

        * **Multivariate (two columns) using Scikit-Learn KNN model with NaNs removed, with k=5 nearby neighbors**
            * Comparison of Results (of "price" vs "predicted_price"):
                * Model Trained with two features (both "accommodates" and "bedrooms" columns):
                    * MAE (Two features): 31.20
                    * MSE (Two features): 13315.05
                    * RMSE (Two features): 115.39
                    * MAE to RMSE Ratio (Two features): 0.27:1
                * Key Changes:
                    * Train using two features/columns (multivariate) instead of just one (univariate)
                    with Scikit-Learn library instead of manually computation
                    * Removal of columns with >20% of its rows being NaN
                    * Columns with <1% of NaNs having the shared row/observation were removed

        * **Multivariate (four columns) using Scikit-Learn KNN model with NaNs removed, with k=5 nearby neighbors**
            * Comparison of Results (of "price" vs "predicted_price"):
                * Model Trained with Four features ("accommodates", "bedrooms", "bathrooms", and "number_of_reviews" columns):
                    * MAE (Four features): 32.90
                    * MSE (Four features): 12754.54
                    * RMSE (Four features): 112.94
                    * MAE to RMSE Ratio (Four features): 0.29:1

            * Screenshots:

                ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/rental_property_listing/evaluation_training_columns.png)

        * **Multivariate (ALL columns) using Scikit-Learn KNN model with NaNs removed, with k=5 nearby neighbors**
            * Comparison of Results (of "price" vs "predicted_price"):
                * Model Trained with ALL features (excluding those containing "id", "_id", or "-id":
                    * 'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'guests_included',
                    'minimum_nights', 'maximum_nights', 'availability_30', 'availability_60', 'availability_90',
                    'availability_365', 'number_of_reviews', 'calculated_host_listings_count
                    * MAE (Multiple features): 30.10
                    * MSE (Multiple features): 11630.41
                    * RMSE (Multiple features): 107.84
                    * MAE to RMSE Ratio (Multiple features): 0.28:1

        * **Hyperparameter Optimisation `k` Results**
            * Hyperparameter Range: 1 to 20
                * Features: ["accommodates", "bedrooms", "bathrooms", and "number_of_reviews"]

                    * Screenshots:

                        ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/rental_property_listing/hyperparameter_1_to_20_four_features.png)

                * Features: All possible combinations of the features with no repeat combinations were compared and plotted (see below):
                    * Best feature combo is 'bedrooms__bathrooms__number_of_reviews' having lowest MSE of 10606.68 using 'k' nearest neighbors of 6 (optimum)

                    * Screenshots:

                        ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/rental_property_listing/hyperparameter_vs_feature_combos.png)

#### Example 2: Car Listing Dataset

* Problem for Dataset 2:
    * Problem (abstract) - How do the properties of a car impact it's price?
    * Problem (narrow) - How does the number of number of doors, curb weight, horsepower, city fuel usage, and highway fuel usage affect a car's price?

* Setup used:
    * Training columns (features): "num-of-doors", "curb-weight", "horsepower", "city-mpg", "highway-mpg"
    * Target column: "price"
    * K-Folds for Cross Validation: 10
    * Hyperparameter k Range: 0 to 20

* **Logistic Regression**

    * **WARNING** Same warning as Example 1

* **Linear Regression**

    * Key Results:
        * **Best Feature Combo**: "num-of-doors", "curb-weight"
        * **Best RMSE**:
            * 4235.36
        * **Pre-Optimisation Best RMSE**:
            * 4206.44
        * **Hyperparameter k of best**:
            * 1

    * Screenshots:

        * Visual evaluation

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/car_listing/evaluation_training_columns.png)

        * Hyperparameter k Optimisation using Linear Regression

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/car_listing/linear_regression.png)

* **KNN Regression**

    * Key Results:
        * **Best Feature Combo**: "num-of-doors", "curb-weight", "city-mpg"
        * **Best RMSE**:
            * 3256.67
        * **Hyperparameter k of best**:
            * 3

    * Screenshots:

        * Hyperparameter k Optimisation using KNN Regression

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/car_listing/knn_regression.png)

#### Example 3: Car Listing "Fuel" Dataset

* Problem for Dataset 3:
    * Problem (abstract) - How do the properties of a car impact it's fuel efficiency?
    * Problem (narrow) - How does the number of cylinders, displacement, horsepower, weight, acceleration, and model year affect a car's fuel efficiency?

* Setup used:
    * Training columns (features): mpg,cylinders,displacement,horsepower,weight,acceleration,model-year,origin,car-name
    * Target column: mpg
    * K-Folds for Cross Validation: 10
    * Hyperparameter k Range: 0 to 20

* **Logistic Regression**

    * **WARNING** Same warning as Example 1 and 2

* **Linear Regression**

    * Key Results:
        * **Best Feature Combo**: weight, model-year, origin
        * **Best RMSE** (to gain quantitative understanding of Models error (mismatch between the Model's Predictions and Actual values):
            * 3.30  (Note: MAE to RMSE Ratio using Linear Regression: 0.61:1)
        * **Hyperparameter k of best**:
            * 1

    * Screenshots:

        * Visual evaluation
        (i.e. evaluate visually the Predicted Target value from training on known features vs Actual Target value to understand Linear Regression Model performance)

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/car_listings_fuel/evaluation_training_columns.png)

        * Hyperparameter k Optimisation using Linear Regression

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/car_listings_fuel/linear_regression.png)

* **KNN Regression**

    * Key Results:
        * **Best Feature Combo**: displacement, weight, model-year
        * **Best RMSE**:
            * 2.75
        * **Hyperparameter k of best**:
            * 9

    * Screenshots:

        * Hyperparameter k Optimisation using KNN Regression
        (i.e. plotting RMSE for different combinations of feature columns to find lowest using KNN Regression)

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/car_listings_fuel/knn_regression.png)

#### Example 4: University Admissions Dataset

* Setup Configuration:
    * Check that the following property values are assigned in `input_event.py`:
        ```
        "model_workflow_config": {
            "model_workflow_for_knn_algorithm": "scikit",
            "model_workflow_for_linear_regression_algorithm_toggle": True,
            "model_workflow_for_logistic_regression_algorithm_toggle": True,
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
        "dataset_selected": "university-admissions",
        "dataset_config": {

            ...

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
            }
        }
        ```

* Setup used:
    * Training columns (features): gpa, gre
    * Target column: admit
    * K-Folds for Cross Validation: 10
    * Hyperparameter k Range: 0 to 20

* **Logistic Regression**

    * Key Results:
        * **Best Feature Combo**: gpa, gre
        * **Best RMSE**:
            * 0.382  (Note: MAE to RMSE Ratio using Logistic Regression: 0.61:1)
        * **Hyperparameter k of best**:
            * 1
        * **Accuracy** of Predictions (Predicted Target value when compared against Actual Target value) using default Discrimination Threshold of 0.5:
            * 0.782 (78.2%)
        * **AUC (Area Under Curve) Score**
            * 0.857 (85.7%)
            * Note: Predicts probability the Classifier will rank a random positive observation higher than a random negative observation.
            If it's higher than 50% (corresponding to simply randomly guessing) then its a good result but may indicate training more features to predict may be necessary.
        * **Sensitivity** of Predictions (using Binary Classification):
            * 0.5
        * **Specificity** of Predictions (using Binary Classification):
            * 0.5

    * Screenshots:

        * Receiver Operator Characteristic (ROC) Curve with AUC Score

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/university_admission/logistic_regression_roc_curve_auc_score.png)

        * Visual evaluation

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/university_admission/evaluation_logistic_regression.png)

        * Hyperparameter k Optimisation using Logistic Regression

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/university_admission/logistic_regression_final.png)

* **Linear Regression**

    * Key Results:
        * **Best Feature Combo**: gpa, gre
        * **Best RMSE**:
            * 0.392  (Note: MAE to RMSE Ratio using Linear Regression: 0.80:1)
        * **Hyperparameter k of best**:
            * 1

    * Screenshots:

        * Visual evaluation

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/university_admission/evaluation_linear_regression.png)

        * Hyperparameter k Optimisation using Linear Regression

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/university_admission/linear_regression_final.png)

* **KNN Regression**

    * Key Results:
        * **Best Feature Combo**: gpa, gre
        * **Best RMSE**:
            * 0.377 with 'k' of 20 (optimum) using 10 K-Folds for Cross Validation
        * **Hyperparameter k of best**:
            * 20

    * Screenshots:

        * Hyperparameter k Optimisation using KNN Regression

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/university_admission/knn_regression_final.png)

#### Example 5: Senator Votes Dataset

* Problem for Dataset 5:
    * Problem (abstract) - What new legislation (decided by roll call votes to pass a bill by Senators) are most likely to cause 'extremist' Senators?
    (whereby some 'extremist' Senators decide not to vote along party lines in accordance with how their political party votes)

* Note: **K-Means Clustering** is used and generates a new **"extremism"** column in the main DataFrame that is subsequently used
by the Prediction Algorithms as the Target Column (but does not exist prior).
Prior to use of K-Means Clustering an **"affiliation_column"** property must be added within the dataset key
and assigned with the column that is associated with each cluster.

* Setup Configuration:
    * Check that the following property values are assigned in `input_event.py`:
        ```
        "model_workflow_config": {
            "model_workflow_for_knn_algorithm": "scikit",
            "model_workflow_for_linear_regression_algorithm_toggle": True,
            "model_workflow_for_logistic_regression_algorithm_toggle": False,
        },
        "training_config": {
            "min_training_features": 0
        },
        "cleansing_config": {
            "min_percentage_incomplete_observations_to_remove_column": 0.2,
            "max_percentage_incomplete_observations_to_retain_column_and_remove_incomplete_slice": 0.02
        },
        "hyperparameter_optimisation_config": {
            "hyperparameter_optimisation_toggle": False,
            "hyperparameter_range": 0,
            "hyperparameter_quantity_fixed": 0
        },
        "k_means_clustering_config": {
            "k_means_clustering_toggle": True,
            "centroids_quantity": 2
        },
        "k_fold_cross_validation_config": {
            "k_fold_cross_validation_toggle": False,
            "k_folds_quantity": 0,
            "k_folds_workflow": "scikit"
        },
        "plot_config": {
            "plot_individual_train_features_vs_target_toggle": False,
            "plot_linear_relationship_prediction_vs_actual_for_train_features_vs_target_toggle": False,
            "plot_logistic_relationship_prediction_vs_actual_for_train_features_vs_target_toggle": False,
            "plot_logistic_roc": False,
            "plot_hyperparameter_optimisation": False
        },
        "dataset_selected": "senators-vote",
        "dataset_config": {

            ...

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
                "training_columns": [],
                "target_column": "00001",
                "cleanse_columns_price_format": [],
                "convert_columns_words_to_digits": []
            }
        ```

* Setup used:
    * Training columns (features): vote-bill1, vote-bill4, vote-bill5, vote-bill6, vote-bill7, vote-bill8
    * Target column: extremism
    * Affiliation column: party
    * K-Folds for Cross Validation: 10
    * Hyperparameter k Range: 0 to 20

* **Linear Regression**

    * Screenshots:

        * Visual evaluation

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/senators_vote/evaluation_linear_regression.png)

        * Hyperparameter k Optimisation using Linear Regression

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/senators_vote/linear_regression.png)

* **KNN Regression**

    * Screenshots:

        * Hyperparameter k Optimisation using KNN Regression

            ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/senators_vote/knn_regression.png)

