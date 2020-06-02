
import pandas as pd
def concatenate(dfs):
    """
    From https://stackoverflow.com/questions/45639350/retaining-categorical-dtype-upon-dataframe-concatenation
    Concatenate while preserving categorical columns.
    NB: We change the categories in-place for the input dataframes"""
    from pandas.api.types import union_categoricals
    import pandas as pd
    # Iterate on categorical columns common to all dfs
    for col in set.intersection(
        *[set(df.select_dtypes(include='category').columns)
            for df in dfs]):
        # Generate the union category across dfs for this column
        uc = union_categoricals([df[col] for df in dfs])
        # Change to union category for all dataframes
        for df in dfs:
            df[col] = pd.Categorical( df[col], categories=uc.categories )
    return pd.concat(dfs)


def dask_ddf_to_df(ddf, partitions_to_concat=10):
    """
    Load and append to Pandas dataframe
    """
    
    dfs = []
    for i in range(partitions_to_concat):
        ddf_partition = ddf.get_partition(i)
        df_temp = ddf_partition.compute()
        dfs.append(df_temp)

    return concatenate(dfs)


def dask_Xy_to_df(ddf, y_ddf, target, partitions_to_concat=10):
    """
    Load and append to Pandas dataframe from X and y ddfs
    """
    ddf[target] = y_ddf[target]
    
    dfs = dask_ddf_to_df(ddf, partitions_to_concat)

    y_df = dfs[target]
    X_df = dfs.drop(columns=target)
    
    # Undoes the adding of the target to the initial ddf
    ddf = ddf.drop(columns = target, axis=1)
    
    return X_df, y_df