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


        
        