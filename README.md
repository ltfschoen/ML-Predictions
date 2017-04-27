---
Machine Learning - Predictions
---

# Table of Contents
  * [Chapter 0 - Results](#chapter-0)
  * [Chapter 1 - Initial Setup](#chapter-1)
  * [Chapter 2 - Known Bugs](#chapter-2)
  * [Chapter 3 - Predict best rental price given data set of other listings](#chapter-3)

## Chapter 0 - Results

* Screenshots

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/screenshot.png)

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/screenshot2.png)

* Comparison of Results (of "price" vs "predicted_price"):
    * Model Trained #1 "accommodates" column:
        * MAE: 58.77 (or ~56.29 without randomising)
        * MSE: 19458.02 (or ~18646.50 without randomising) (i.e. $ squared, penalises predictions further from actual)
    * Model Trained #2 "bathrooms" column:
        * MAE: 58.77
        * MSE: 16233.52 (or ~17333.4 without randomising)

## Chapter 1 - Initial setup <a id="chapter-1"></a>

* Install dependencies:
    ```
    pip3 install pandas Path matplotlib requests numpy math
    ```

* Setup image rendering [backend](http://matplotlib.org/faq/usage_faq.html#what-is-a-backend) of matplotlib on macOS:
    ```
    touch ~/.matplotlib/matplotlibrc; echo 'backend: TkAgg' >> ~/.matplotlib/matplotlibrc`
    ```

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
    ```
    python3 prediction.py
    ```

* Note: Change from `np.random.seed(1)` to `np.random.seed(0)` to generate different instead of
same random permutations each time its run.