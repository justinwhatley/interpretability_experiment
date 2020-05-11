def normalize_df_cont(df, continuous_columns):
    """
    Normalize the continuous colums in a dataframe  
    """
    means, stds = {},{}
    for column in continuous_columns:
        means[column], stds[column] = df[column].mean(), df[column].std()
        df[column] = (df[column]-means[column]) / (1e-7 + stds[column])


def normalize_ddf_cont(ddf, continuous_columns):
    """
    TODO
    Normalize the continuous colums in a dask dataframe  
    """
    means, stds = {},{}
    for column in continuous_columns:
        means[column], stds[column] = ddf[column].mean(), ddf[column].std()
        ddf[column] = (ddf[column]-means[column]) / (1e-7 + stds[column])


