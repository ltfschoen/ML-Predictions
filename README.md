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

* Problem for Dataset 1:
    * Problem (abstract) - What's the optimum nightly price (prediction) to rent a living space using existing data on local listings?
    * Problem (narrow) - What's the optimum nightly price (prediction) to rent a living space using existing data of on local listings based on their attributes including price, number of bedrooms, room type?

* Problem for Dataset 2:
    * Problem (abstract) - What's the optimum nightly price (prediction) to rent a living space using existing data of local listings?
    * Problem (narrow) - What's the optimum price to sell a car (prediction) using existing data of sales prices and their attributes including make, fuel type?

* Problem for Dataset2:
    * Problem (abstract) - How do the properties of a car impact it's fuel efficiency?
    * Problem (narrow) - How does the number of cylinders, displacement, horsepower, weight, acceleration, and model year affect a car's fuel efficiency?

* Applies **machine learning** techniques on past data using a **regression** process since
we want to predict a specific optimum number value for a chosen feature (column) of a given dataset based on
chosen list of features to train against by using a chosen Train/Test Validation Process with
the K-Nearest-Neighbors Machine Learning Model such as that provided by the Scikit-Learn library,
which uses Similarity Metrics such as the Euclidean Distance equation to perform comparisons,
and evaluate the quality of the prediction accuracy using Error Metrics such as
Median Average Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) equations,
where RMSE penalises large errors more (caused by outliers). Refer to Implementation Guide
in `prediction_config.py` for details.

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
* Configure desired input values into EVENT object within `input_event.py`
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

* Note that the Error Metrics reduce in multivariate configurations where more columns
are used to train the KNN model and when rows with NaN values are removed. Adding all columns as features
instead of just the best four features actually reduces accurancy.
The RMSE indicates how inaccurate we expect the predicted price value to be on average
(i.e. if RMSE is $127 its means the model inaccurately predicts the price values by $127 on average,
which means the model's usefulness must be improved by reducing the RMSE particularly when the
average value in the "price" column is $300)

* **MAE to RMSE Ratio Usage**
    * Compare MAE to RMSE Ratio to highlight if "outliers" exist that cause large but infrequent errors.
    * Expect MAE > RMSE (since RMSE takes the square root of the squared error MAE)
    * RMSE penalises large errors more than MAE
    * RMSE is in the denominator of the ratio such that higher RMSE results in smaller ratio.
    * Expect a model to contain large outliers when MAE < RMSE

### Result Statistics and Screenshots

#### Example 1: Rental Property Listing Dataset

* **KNN Regression**

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

    * **K-Fold Cross Validation Results** (with Hyperparam 'k' Optimisation)
        * Features:
        * MSE
            * Best feature combo 'bedrooms', 'bathrooms', 'number_of_reviews'
            * Best MSE result **improved robustness of results since takes average MSE over K-Folds**:
                * 12718.88 with 'k' of 17 (optimum) using 10 K-Folds for Cross Validation
        * RMSE
            * Best feature combo 'accommodates', 'bedrooms', 'bathrooms', and 'number_of_reviews'
            * Best RMSE results (**improved robustness of results since takes average RMSE over K-Folds**):
                * 113.96 with 'k' of 20 (optimum) using 3 K-Folds for Cross Validation
                * 109.73 with 'k' of 20 (optimum) using 5 K-Folds for Cross Validation
                * 108.60 with 'k' of 19 (optimum) using 10 K-Folds for Cross Validation

                * Screenshots:

                    ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/rental_property_listing/rmse_cross_validation_fixed2.png)

#### Example 2: Car Listing Dataset

* Summary of results:
    * Training features: "num-of-doors", "curb-weight", "horsepower", "city-mpg", "highway-mpg"
    * Target column: "price"
    * RMSE results:
        * Best feature combo: "num-of-doors", "curb-weight", "city-mpg"
        * Best RMSE using **KNN Regression** with **K-Fold Cross Validation** and Hyperparam 'k' Optimisation:
            * 3256.67 with 'k' of 3 (optimum) using 10 K-Folds for Cross Validation

* **Linear Regression**

    ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/car_listing/evaluation_training_columns.png)

* **KNN Regression**

    ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/car_listing/rmse_cross_validation_fixed2.png)

#### Example 3: Car Listing "Fuel" Dataset

* Summary of results:
    * Training features: mpg,cylinders,displacement,horsepower,weight,acceleration,model-year,origin,car-name
    * Target column: mpg
    * RMSE results:
        * Best feature combo: displacement, weight, model-year
        * Best RMSE (Error Metric) using **Linear Regression**
        (to gain quantitative understanding of Models error (mismatch between the Model's Predictions and Actual values):
            * 3.30  (Note: MAE to RMSE Ratio using Linear Regression: 0.61:1)
        * Best RMSE using **KNN Regression** with **K-Fold Cross Validation** and Hyperparam 'k' Optimisation:
            * 2.75 with 'k' of 9 (optimum) using 10 K-Folds for Cross Validation

* **Linear Regression**
    * Evaluate visually the Predicted Target value from training on known features vs Actual Target value
    to understand Linear Regression Model performance

    ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/car_listings_fuel/evaluation_training_columns.png)

* **KNN Regression**
    * Plotting RMSE for different combinations of feature columns to find lowest using KNN Regression

    ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/car_listings_fuel/knn_rmse_cross_validation_fixed.png)

#### Example 4: University Admissions Dataset

* Summary of results:
    * Training features: gpa, gre
    * Target column: admit
    * **Accuracy** of Predictions (Predicted Target value when compared against Actual Target value):
        * Accuracy using **Logistic Regression** with Discrimination Threshold of 0.5:
            * 0.782 (78.2%)
    * **AUC (Area Under Curve) Score** that predicts the probability that the Classifier
    will rank a random positive observation higher than a random negative observation.
    If it's higher than 50% (corresponding to simply randomly guessing) then its a good result
    but may indicate training more features to predict may be necessary.
        * 0.857 (85.7%)
    * **Sensitivity** of Predictions (using Binary Classification):
        * 0.5
    * **Specificity** of Predictions (using Binary Classification):
        * 0.5
    * **Error Metric** (RMSE) results:
        * Best feature combo: gpa, gre
        * Best RMSE using **Linear Regression**:
            * 0.392  (Note: MAE to RMSE Ratio using Linear Regression: 0.80:1)
        * Best RMSE using **Logistic Regression**:
            * 0.382  (Note: MAE to RMSE Ratio using Logistic Regression: 0.61:1)
        * Best RMSE using **KNN Regression** with **K-Fold Cross Validation** and Hyperparam 'k' Optimisation:
            * 0.377 with 'k' of 20 (optimum) using 10 K-Folds for Cross Validation

MAE: 0.23304931819072985
MSE: 0.14644020478518427
RMSE: 0.3826750642322861
MAE to RMSE Ratio using Logistic Regression: 0.61:1

* **Linear Regression**

    ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/university_admission/linear_regression.png)

* **Logistic Regression**

    * Receiver Operator Characteristic (ROC) Curve with AUC Score

        ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/university_admission/logistic_regression_roc_curve_auc_score.png)

    * Evaluate visually the Predicted Target value from training on known features vs Actual Target value
    to understand Logistic Regression Model performance

        ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/university_admission/logistic_regression.png)

* **KNN Regression**

    ![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/university_admission/knn_regression_fixed.png)
