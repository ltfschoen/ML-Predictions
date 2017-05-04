---
Machine Learning - Predictions
---

# Table of Contents
  * [Chapter 0 - About](#chapter-0)
  * [Chapter 1 - Setup](#chapter-1)
  * [Chapter 2 - Results](#chapter-2)
  * [Chapter 3 - Known Bugs](#chapter-3)


## Chapter 0 - About <a id="chapter-0"></a>

* Predict the optimum value for a chosen feature (column) of a given dataset based on
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

* Read the Implementation Guide in `prediction_config.py` and configure desired values.

* Run
    ```
    python3 main.py
    ```

* Note: Change from `np.random.seed(1)` to `np.random.seed(0)` to generate different instead of
same random permutations each time its run.

## Chapter 2 - Results <a id="chapter-2"></a>

### Summary of Results

* Note that the Error Metrics reduce in multivariate configurations where more columns
are used to train the KNN model and when rows with NaN values are removed. The
RMSE indicates how inaccurate we expect the predicted price value to be on average
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

* **Univariate (one column) using manual KNN model, with k=5 nearby neighbors**
    * Comparison of Results (of "price" vs "predicted_price"):
        * Model Trained #1 "accommodates" column:
            * MAE: 58.77 (or ~56.29 without randomising)
            * MSE: 19458.02 (or ~18646.50 without randomising) (i.e. $ squared, penalises predictions further from actual)
            * RMSE: 139.49
            * MAE to RMSE Ratio: 0.42:1
        * Model Trained #2 "bathrooms" column:
            * MAE: 58.77
            * MSE: 16233.52 (or ~17333.4 without randomising)
            * RMSE: 127.37 (or 131.66 without randomising)
            * MAE to RMSE Ratio: 0.46:1

* Screenshots:

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part1/screenshot_accommodates_feature_univariate.png)

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part1/screenshot_bedrooms_feature_univariate.png)

* **Univariate (one column) using manual KNN model with NaNs removed, with k=5 nearby neighbors**
    * Comparison of Results (of "price" vs "predicted_price"):
        * Model Trained #1 "accommodates" column:
            * MAE: 53.46
            * MSE: 16208.90
            * RMSE: 127.31
            * MAE to RMSE Ratio: 0.42:1
        * Model Trained #2 "bathrooms" column:
            * MAE: 58.85
            * MSE: 15423.09
            * RMSE: 124.19
                * Note: Expect model to be off by $124 on average for predicted price values
            * MAE to RMSE Ratio: 0.47:1
        * Key Changes:
            * Removal of columns with >20% of its rows being NaN
            * Columns with <1% of NaNs having the shared row/observation were removed

* Screenshots:

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part2/screenshot_accommodates_feature_univariate_post_strip_and_normalisation.png)

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part2/screenshot_bedrooms_feature_univariate_post_strip_and_normalisation.png)

* **Multivariate (two columns) using Scikit-Learn KNN model with NaNs removed, with k=5 nearby neighbors**
    * Comparison of Results (of "price" vs "predicted_price"):
        * Model Trained with two features (both "accommodates" and "bedrooms" columns):
            * MAE (Two features): 33.80
            * MSE (Two features): 12621.33
            * RMSE (Two features): 112.34
            * MAE to RMSE Ratio (Two features): 0.30:1
        * Key Changes:
            * Train using two features/columns (multivariate) instead of just one (univariate)
            with Scikit-Learn library instead of manually computation
            * Removal of columns with >20% of its rows being NaN
            * Columns with <1% of NaNs having the shared row/observation were removed

* Screenshots:

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part3/screenshot_accommodates_feature_multivariate_post_strip_and_normalisation.png)

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part3/screenshot_bedrooms_feature_multivariate_post_strip_and_normalisation_fix.png)

* **Multivariate (four columns) using Scikit-Learn KNN model with NaNs removed, with k=5 nearby neighbors**
    * Comparison of Results (of "price" vs "predicted_price"):
        * Model Trained with Four features ("accommodates", "bedrooms", "bathrooms", and "number_of_reviews" columns):
            * MAE (Four features): 30.599999999999994
            * MSE (Four features): 10859.249674556213
            * RMSE (Four features): 104.20772367994712
            * MAE to RMSE Ratio (Four features): 0.29:1

* Screenshots:
    * Note: First two column plots for "accommodates" and "bedrooms" were the same as the Multivariate (two columns) plots

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part3/screenshot_bathrooms_feature_multivariate_post_strip_and_normalisation.png)

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part3/screenshot_number_of_reviews_feature_multivariate_post_strip_and_normalisation.png)

* **Multivariate (ALL columns) using Scikit-Learn KNN model with NaNs removed, with k=5 nearby neighbors**
    * Comparison of Results (of "price" vs "predicted_price"):
        * Model Trained with ALL features (excluding those containing "id", "_id", or "-id":
            * 'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'guests_included',
            'minimum_nights', 'maximum_nights', 'availability_30', 'availability_60', 'availability_90',
            'availability_365', 'number_of_reviews', 'calculated_host_listings_count
            * MAE (Multiple features): 24.400000000000006
            * MSE (Multiple features): 8049.1010798816569
            * RMSE (Multiple features): 89.71678259880733
            * MAE to RMSE Ratio (Multiple features): 0.27:1

## Chapter 3 - Known Bugs <a id="chapter-3"></a>

* Warning occurs:
    ```
    SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    ```
