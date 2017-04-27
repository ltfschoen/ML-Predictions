---
Machine Learning - Predictions
---

# Table of Contents
  * [Chapter 1 - Initial setup](#chapter-1)
  * [Chapter 2 - Initial setup](#chapter-2)
  * [Chapter 3 - Predict best rental price given data set of other listings](#chapter-3)

## Chapter 0 - Results

* Screenshots

![alt tag](https://raw.githubusercontent.com/ltfschoen/ML-Predictions/master/screenshots/screenshot.png)

## Chapter 1 - Initial setup <a id="chapter-1"></a>

## Chapter 2 - Known Bugs <a id="chapter-2"></a>

    * Warning occurs:
        ```
        SettingWithCopyWarning:
        A value is trying to be set on a copy of a slice from a DataFrame.
        Try using .loc[row_indexer,col_indexer] = value instead

        See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
        ```

## Chapter 3 - Predict best rental price given data set of other listings <a id="chapter-3"></a>

`python3 predict_rental_list_price.py`

* Note: Change from `np.random.seed(1)` to `np.random.seed(0)` to generate different instead of
same random permutations each time its run.