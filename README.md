---
Machine Learning - Predictions
---

# Table of Contents
  * [Chapter 0 - Results](#chapter-0)
  * [Chapter 1 - Initial Setup](#chapter-1)
  * [Chapter 2 - Known Bugs](#chapter-2)
  * [Chapter 3 - Predict best rental price given data set of other listings](#chapter-3)

## Chapter 0 - Results

* [1]:
    * Removal of columns with >20% of its rows being NaN
    * Columns with <1% of NaNs having the shared row/observation were removed

* [2]:
    * Train using two features/columns (multivariate) instead of just one (univariate) with Scikit-Learn library instead of manually computation

* Screenshots (**BEFORE** [1] and [2] occurred):

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part1/screenshot_accommodates_feature_univariate.png)

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part1/screenshot_bedrooms_feature_univariate.png)

* Comparison of Results (of "price" vs "predicted_price"), **BEFORE** [1] occurred to reduce error:
    * Model Trained #1 "accommodates" column:
        * MAE: 58.77 (or ~56.29 without randomising)
        * MSE: 19458.02 (or ~18646.50 without randomising) (i.e. $ squared, penalises predictions further from actual)
        * RMSE: 139.49
        * MAE to RMSE Ratio: 0.42:1
    * Model Trained #2 "bathrooms" column:
        * MAE: 58.77
        * MSE: 16233.52 (or ~17333.4 without randomising)
        * RMSE: 127.37 (or 131.66 without randomising)
            * Note: Expect model to be off by $127 on average for predicted price values
        * MAE to RMSE Ratio: 0.46:1
    * **MAE to RMSE Ratio**
        * Definition: Comparing MAE to RMSE Ratio helps highlight if "outliers" exist that cause large but infrequent errors.
        Given that we expect MAE > RMSE (since RMSE takes the square root of the squared error MAE)
        Given that RMSE penalises large errors more than MAE,
        Given that RMSE is in the denominator of the ratio such that higher RMSE results in smaller ratio
        Conclude that both models contain large outliers (since MAE < RMSE)
        Conclude that "bathrooms" model with higher ratio performs with better accuracy than "accommodates" model
    * Important Note:
        * Most rental listings are listed at ~$300 so must reduce RMSE error to improve model usefulness

* Screenshots (**AFTER** [1] occurred):

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part2/screenshot_accommodates_feature_univariate_post_strip_and_normalisation.png)

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part2/screenshot_bedrooms_feature_univariate_post_strip_and_normalisation.png)

* Comparison of Results (of "price" vs "predicted_price"), **AFTER** [1] occurred to reduce error:
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

* Screenshots Multivariate Two Columns (**AFTER** [1] and [2] occurred):

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part3/screenshot_accommodates_feature_multivariate_post_strip_and_normalisation.png)

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part3/screenshot_bedrooms_feature_multivariate_post_strip_and_normalisation_fix.png)

* Comparison of Results (of "price" vs "predicted_price"), **AFTER** [1] and [2] occurred to reduce error:
    * Model Trained with two features (both "accommodates" and "bedrooms" columns):
        * MAE (Two features): 33.80
        * MSE (Two features): 12621.33
        * RMSE (Two features): 112.34
        * MAE to RMSE Ratio (Two features): 0.30:1

* Screenshots Multivariate Four Columns (**AFTER** [1] and [2]). Note: First two plots shown above

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part3/screenshot_bathrooms_feature_multivariate_post_strip_and_normalisation.png)

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/part3/screenshot_number_of_reviews_feature_multivariate_post_strip_and_normalisation.png)

* Comparison of Results (of "price" vs "predicted_price"), **AFTER** [1] and [2] occurred to reduce error:
    * Model Trained with Four features ("accommodates", "bedrooms", "bathrooms", and "number_of_reviews" columns):
        * MAE (Four features): 30.599999999999994
        * MSE (Four features): 10859.249674556213
        * RMSE (Four features): 104.20772367994712
        * MAE to RMSE Ratio (Four features): 0.29:1

* Comparison of Results (of "price" vs "predicted_price") **AFTER** [1] and [2] occurred to reduce error:
    * Model Trained with ALL features (excluding those containing "id", "_id", or "-id":
        * 'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'guests_included',
        'minimum_nights', 'maximum_nights', 'availability_30', 'availability_60', 'availability_90',
        'availability_365', 'number_of_reviews', 'calculated_host_listings_count
        * MAE (Multiple features): 24.400000000000006
        * MSE (Multiple features): 8049.1010798816569
        * RMSE (Multiple features): 89.71678259880733
        * MAE to RMSE Ratio (Multiple features): 0.27:1

## Chapter 1 - Initial setup <a id="chapter-1"></a>

* Install dependencies:
    ```
    pip freeze > requirements.txt
    pip install -r requirements.txt
    ```

* Setup image rendering [backend](http://matplotlib.org/faq/usage_faq.html#what-is-a-backend) of matplotlib on macOS:
    ```
    touch ~/.matplotlib/matplotlibrc; echo 'backend: TkAgg' >> ~/.matplotlib/matplotlibrc`
    ```

* Configure version to run in `main.py`:
    * Manual Machine Learning algorithm
    * Scikit-Learn Machine Learning algorithm
        * Important Note: If NaN values found prevent processing, check the percentage of
        NaN rows in columns being trained and increase value of MAX_MINOR_INCOMPLETE above that percentage

* Run
    ```
    python3 main.py
    ```

* Note: Change from `np.random.seed(1)` to `np.random.seed(0)` to generate different instead of
same random permutations each time its run.

## Chapter 2 - Known Bugs <a id="chapter-2"></a>

* Warning occurs:
    ```
    SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    ```

## Chapter 3 - Predict best rental price given data set of other listings <a id="chapter-3"></a>

* Given you have a rental listing that accommodates up to 3 rooms.
And given a data set that contains features (column attributes) of other rental listings.
Find the optimum rental listing price using similarity metrics
