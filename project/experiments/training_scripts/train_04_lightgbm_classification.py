import lightgbm as lgb




def lightgbm_preprocess(X_ddf, y_ddf, target, part_samples):
    
    X, y = convert_to_pandas(X_train, y_train, target, part_samples)

    return X, y


def train_lightgbm(lgb_train, lgb_eval, lgb_params, save_to, recompute=False):
    """
    In memory training of lightGBM
    """
    import joblib
    if not recompute:
        try:
            print('Loading previously trained model: ' + str(save_to))
            return joblib.load(save_to)
        except:
            print('Model not found')
    
    from sklearn.linear_model import LogisticRegression

    print('Training model')
    lgb_model = lgb.train(lgb_params, 
                      lgb_train, 
                      10000, 
                      valid_sets=lgb_eval, 
                      early_stopping_rounds=50, 
                      verbose_eval=100)
    
    print('Saving model to: ' + str(save_to))
    joblib.dump(estimator, save_to)
    
    return lgb_model


def get_categorical_indices(ddf):
    """
    Gets categorical column indices - may not be necessary for Pandas, but doesn't hurt
    """
    categorical_feature_names = ddf.select_dtypes(['category']).columns
    categorical_feature_columns = [ddf.columns.get_loc(x) for x in categorical_feature_names]
    return categorical_feature_columns


def train_model_lightgbm(X, y, lgb_params, save_to, recompute=False):
    # TODO figure out how to train with incremental learning
        
    categorical_col_indices = get_categorical_indices(X)
    
    # Make training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8)

    
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature = categorical_col_indices)
    lgb_eval = lgb.Dataset(X_test, y_test, categorical_feature = categorical_col_indices)

    #https://www.kaggle.com/mlisovyi/beware-of-categorical-features-in-lgbm
    # https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
    lgb_model = lgb.train(lgb_params,
                          lgb_train,
                          num_boost_round=10000,
                          valid_sets=lgb_eval,
                          verbose_eval=100,
                          early_stopping_rounds=500, 
                         )
    return lgb_model