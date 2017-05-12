import pandas as pd

def build_folds(df, k_folds):
    """ Generate indexes of lower and upper extents of each fold for given by k-folds value

    Given k-folds 3 and DataFrame length 7 it returns [0, 2, 4, 7].
    Example usage is to set lower and upper indexes in "fold" column of DataFrame
    for assignment of fold values
        df.set_value(df.index[0:2], "fold", 1)
        df.set_value(df.index[2:4], "fold", 2)
        df.set_value(df.index[4:7], "fold", 3)
    """
    min_k_folds = 2
    from_index = 0

    if k_folds < min_k_folds:
        folds = [from_index, len(df)]
    else:
        to_index_init = int(round((len(df) / k_folds), 0)) # Round up
        folds = [from_index, to_index_init]

        def get_next_index():
            return folds[-1] + to_index_init

        i = 2 # Start at fold no. 2 since already assigned fold no. 1 to `folds`
        while get_next_index() < len(df) and i < k_folds:
            next_index = folds[-1] + to_index_init
            folds.append(next_index)
            i += 1
        folds.append(len(df))
    return folds
