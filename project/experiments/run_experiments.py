from pathlib import Path
import lightgbm as lgb


def load_config(config_path='config.ini'):
    """
    Loads common configuration parameters
    """
    from project.src.utils.configuration_manager import Config
    return Config(config_path)


def init_local_dask(dashboard_address=':20100', memory_limit='4G'):
    """
    Set up local cluster
    """
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(dashboard_address=dashboard_address, memory_limit=memory_limit)
    client = Client(cluster)
    print(client)
    return client


def init_ddf_ohe():
    """
    Dask OHE object, saved for consitency
    """
    from dask_ml.preprocessing import OneHotEncoder
    return OneHotEncoder(sparse=False)


def init_dataset_manager(config):
    """
    Set up dataset manager to handle dataset manipulations 
    """
    from project.src.preprocessing.dataset_manager import DatasetManager
    return DatasetManager(config)


def linear_preprocess(X_ddf, y_ddf, target, encoder, categorical_columns_to_ohe, datetime_columns_to_ohe, part_samples):
    """
    Preprocessing step to transform 
    """
    import project.src.preprocessing.preprocessing_pipelines as preprocessing
    
    
    X_ddf = preprocessing.ohe_preprocessing_pipeline(encoder, 
                                     X_ddf, 
                                     categorical_columns_to_transform = categorical_columns_to_ohe, 
                                     datetime_columns_to_transform = datetime_columns_to_ohe)
    
    X, y = convert_to_pandas(X_ddf, y_ddf, target, part_samples)
    
    fill_nulls(X)

    return X, y


def convert_to_pandas(X_ddf, y_ddf, target, partitions_to_concat):
    """
    Conversion to pandas/moving dataframe to memory for library support
    """
    import project.src.preprocessing.dask_to_pandas as dtp
    X, y = dtp.dask_Xy_to_df(X_ddf, y_ddf, target, partitions_to_concat)
#     print(X.head())
    
    return X, y

def fill_nulls(df):
    
    # Fill null values
#     categorical_cols = X.select_dtypes(include=['category'])
#     for category in categorical_cols:
#         X[category].add_categories('Unknown', inplace=True)
#     X[categorical_cols].fillna('Unknown', inplace=True)
    
    df.fillna(0, inplace=True)


def train_logistic_regression(X, y, save_to, recompute=False):
    """
    In memory training of linear regression
    """
    import joblib
    if not recompute:
        try:
            print('Loading previously trained model: ' + str(save_to))
            return joblib.load(save_to)
        except:
            print('Model not found')
    
    from sklearn.linear_model import LogisticRegression
    
    estimator = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42)
    print('Training model')
    estimator.fit(X, y=y)
    
    print('Saving model to: ' + str(save_to))
    joblib.dump(estimator, save_to)
    
    return estimator



def test_classification(X, y, estimator):
    """
    In memory testing of linear regression 
    """
    from sklearn.metrics import classification_report
    
    predictions = estimator.predict(X)
    print(classification_report(predictions, y))


    
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


if __name__ == "__main__":
    """
    Setup
    """
    # 1. load configuration details
    config = load_config()
    # Set up a local Dask cluster
    dask_client = init_local_dask()
    
    
    """
    Global data preparation
    """
    # 1. get dataset manager to handle dataset manipulations
    dataset_manager = init_dataset_manager(config)

    # 2. convert raw data to parquet (raw -> interim) 
    # 3. get interim data statistics    
    # 4. store statistics
    
    # 5. prepare training and test data sets (interim -> preprocessed)
    dataset_manager.write_dataset(test_size=0.5, overwrite=False)
    
    
    """
    Local data preparation
    """
    partitions_sample = 10
    target = 'Status'
    
    # load test_train datasets (lazy-loading w/ Dask)
    X_train, y_train = dataset_manager.get_training_set()
    X_test, y_test = dataset_manager.get_test_set()
    
    
    recompute = False
    """
    ***********************
    Linear Model Experiment
    ***********************
    """
    # local preprocessing (e.g., OHE for linear models)
    logr_reg_filename = 'logr_estimator_w_cat.sav'
    categorical_columns_to_ohe = ['HourSlot', 'Service']
    datetime_columns_to_ohe = []
    encoder = init_ddf_ohe()
    
    X, y = linear_preprocess(X_train, 
                                y_train,
                                target,
                                encoder, 
                                categorical_columns_to_ohe, 
                                datetime_columns_to_ohe, 
                                partitions_sample)

    """
    Fit models to training data and test against test data
    """
    logistic_reg_save = Path(config.models_directory, logr_reg_filename)    
    logr_estimator = train_logistic_regression(X, y, save_to=logistic_reg_save, recompute=recompute)

    # assess model prediction
    
    X, y = linear_preprocess(X_test, 
                            y_test,
                            target,
                            encoder, 
                            categorical_columns_to_ohe, 
                            datetime_columns_to_ohe, 
                            partitions_sample)
    
    test_classification(X, y, logr_estimator)

    
    """
    Get interpretation
    """
    import eli5
    # apply interpretability strategy
    explanation_df = eli5.explain_weights_df(logr_estimator, feature_names=X.columns.values)
    explanation_df.sort_values('weight', inplace=True)
    print(explanation_df.head())
    
    

    """
    ****************************
    Gradient Boosting Experiment
    ****************************
    """
    
    
    X, y = lightgbm_preprocess(X_train, 
                                y_train,
                                target,
                                partitions_sample)
    
    """
    LightGBM parameters
    """
    params = {
                "max_bin": 512,
                "learning_rate": 0.05,
                "boosting_type": "gbdt",
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": 10, # Low value will decrease likelihood of overfitting
                "verbose": -1,
                "min_data": 100,
                "boost_from_average": True}
    
    """
    Training
    """
    
    lgb_reg_filename = 'lgb_ran_estimator.sav'
    lgb_reg_save = Path(config.models_directory, lgb_reg_filename)

    lgb_model = train_model_lightgbm(X, y, params, lgb_reg_save)
    
    
    """
    Test
    """
    
    X, y = lightgbm_preprocess(X_test, 
                                y_test,
                                target,
                                partitions_sample)
    
    test_classification(X, y, lgb_model)

    



