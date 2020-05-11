def normalize(df, continuous_columns):
    """
    Normalize the continuous colums in a dataframe  
    """
    means, stds = {},{}
    for column in continuous_columns:
        means[column], stds[column] = df[column].mean(), df[column].std()
        df[column] = (df[column]-means[column]) / (1e-7 + stds[column])
