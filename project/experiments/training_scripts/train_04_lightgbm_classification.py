import lightgbm as lgb


LGB_PARAMS = {
            "max_bin": 512,
            "learning_rate": 0.01,
            "boosting_type": "gbdt",
            "objective": "binary",
#             "metric": "binary_logloss",
            "metric": "auc",
            "num_leaves": 10, # Low value will decrease likelihood of overfitting
            "verbose": -1,
            "min_data": 100,
            "boost_from_average": True}


from project.utils.models.model_handling import load_or_store_model
@load_or_store_model
def train_lightgbm_classification(X, y, save_to, recompute=False):    
    """
    In memory training of lightGBM
    """

    X['Status'] = y
    categorical_col_indices = get_categorical_indices(X)
    X.drop(['Status'], axis=1, inplace=True)
    
    # Make training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8)

    
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature = categorical_col_indices)
    lgb_eval = lgb.Dataset(X_test, y_test, categorical_feature = categorical_col_indices)
    
    #https://www.kaggle.com/mlisovyi/beware-of-categorical-features-in-lgbm
    # https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
    lgb_model = lgb.train(LGB_PARAMS,
                          lgb_train,
                          num_boost_round=1000,
                          valid_sets=lgb_eval,
                          verbose_eval=100,
                          early_stopping_rounds=500, 
                         )
    

    return lgb_model



def get_categorical_indices(ddf):
    """
    Gets categorical column indices - may not be necessary for Pandas, but doesn't hurt
    """
    categorical_feature_names = ddf.select_dtypes(['category']).columns
    categorical_feature_columns = [ddf.columns.get_loc(x) for x in categorical_feature_names]
    return categorical_feature_columns




# # https://gist.github.com/goraj/6df8f22a49534e042804a299d81bf2d6
# def train_lightgbm_regression_incremental(X, y, save_to, recompute=False):
#     # TODO figure out how to train with incremental learning
    
#     """
#     In memory training of lightGBM
#     """
#     if not recompute:
#         try:
#             print('Loading previously trained model: ' + str(save_to))
#             return joblib.load(save_to)
#         except:
#             print('Model not found')
        
#     X['Status'] = y
#     categorical_col_indices = get_categorical_indices(X)
#     X.drop(['Status'], axis=1, inplace=True)
    
#     # Make training and validation sets
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8)

    
#     lgb_train = lgb.Dataset(X_train, y_train, categorical_feature = categorical_col_indices)
#     lgb_eval = lgb.Dataset(X_test, y_test, categorical_feature = categorical_col_indices)
    
#     #https://www.kaggle.com/mlisovyi/beware-of-categorical-features-in-lgbm
#     # https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
#     lgb_model = lgb.train(LGB_PARAMS,
#                           lgb_train,
#                           num_boost_round=1000,
#                           valid_sets=lgb_eval,
#                           verbose_eval=100,
#                           early_stopping_rounds=500, 
#                          )
    
#     print('Saving model to: ' + str(save_to))
#     joblib.dump(lgb_model, save_to)
    
#     return lgb_model