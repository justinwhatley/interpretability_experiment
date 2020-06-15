
def datetime_preprocessing_pipeline(ddf, datetime_columns_to_transform = []):
    """
    Preprocessing pipeline that transforms ddf before training
    """
    from project.utils.preprocessing.datetime_to_cat import add_datetime_cat
    
    ddf, new_categorical_columns = add_datetime_cat(ddf, datetime_columns_to_transform)
    ddf = ddf.drop(datetime_columns_to_transform, axis=1)
    
    return ddf


def ohe_preprocessing_pipeline(encoder, ddf, categorical_columns_to_transform = [], datetime_columns_to_transform = []):
    """
    Preprocessing pipeline that transforms ddf before training
    Only keeps floats and the OHE of specified columns to transform
    """
    from project.utils.preprocessing.ohe_ddf_transformer import TransformerOHE
    from project.utils.preprocessing.datetime_to_cat import add_datetime_cat
    
    ddf, new_categorical_columns = add_datetime_cat(ddf, datetime_columns_to_transform)
    categorical_columns_to_transform = categorical_columns_to_transform + new_categorical_columns

    # Get OHE for columns to transform
    transformer = TransformerOHE(ddf, encoder, categorical_columns_to_transform)
    ohe_ddf = transformer.fit_transform()
    
    # Select input columns, here we'll take all floats and ints and OHE columns we prepared
    original_float_columns = list(ddf.select_dtypes(['float']).columns)
    original_int_columns = list(ddf.select_dtypes(['int']).columns)
    original_input_columns = original_float_columns + original_int_columns
    ohe_columns = list(ohe_ddf.columns.values)

    # Remove old categories and update  
    ddf = ddf[original_input_columns]
    ddf[ohe_columns] = ohe_ddf
    
    return ddf


