import lightgbm as lgb
import joblib


LGB_PARAMS = {
            "max_bin": 512,
            "learning_rate": 0.05,
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 10, # Low value will decrease likelihood of overfitting
            "verbose": -1,
            "min_data": 100,
            "boost_from_average": True}



def train_lightgbm_regression(X, y, save_to, recompute=False):
    # TODO figure out how to train with incremental learning
    
    """
    In memory training of lightGBM
    """
    if not recompute:
        try:
            print('Loading previously trained model: ' + str(save_to))
            return joblib.load(save_to)
        except:
            print('Model not found')
        
    X['Status'] = y
    categorical_col_indices = get_categorical_indices(X)
    X.drop(['Status'], axis=1, inplace=True)
    
    # Make training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8)
#     print(X_train.dtypes)
#     print('***')
#     print(y_train.dtypes)
#     print('***')

    
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature = categorical_col_indices)
    lgb_eval = lgb.Dataset(X_test, y_test, categorical_feature = categorical_col_indices)
    print('made it')
    
    #https://www.kaggle.com/mlisovyi/beware-of-categorical-features-in-lgbm
    # https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
    lgb_model = lgb.train(LGB_PARAMS,
                          lgb_train,
                          num_boost_round=1000,
                          valid_sets=lgb_eval,
                          verbose_eval=100,
                          early_stopping_rounds=500, 
                         )
    
    print('Saving model to: ' + str(save_to))
    joblib.dump(lgb_model, save_to)
    
    return lgb_model




def get_categorical_indices(ddf):
    """
    Gets categorical column indices - may not be necessary for Pandas, but doesn't hurt
    """
    categorical_feature_names = ddf.select_dtypes(['category']).columns
    categorical_feature_columns = [ddf.columns.get_loc(x) for x in categorical_feature_names]
    return categorical_feature_columns


# def train_lightgbm_regression(X, y, save_to, recompute=False):
#     """
#     In memory training of lightGBM
#     """
#     if not recompute:
#         try:
#             print('Loading previously trained model: ' + str(save_to))
#             return joblib.load(save_to)
#         except:
#             print('Model not found')
    

#     print('Training model')
#     lgb_model = lgb.train(LGB_PARAMS, 
#                           lgb_train, 
#                           10000, 
#                           valid_sets=lgb_eval, 
#                           early_stopping_rounds=50, 
#                           verbose_eval=100)
    
#     print('Saving model to: ' + str(save_to))
#     joblib.dump(estimator, save_to)
    
#     return lgb_model