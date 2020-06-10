from pathlib import Path
import joblib
recompute=False
cat_columns_to_ohe = ['HourSlot', 'Service']
date_columns_to_ohe = []


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


def init_dataset_manager(config):
    """
    Set up dataset manager to handle dataset manipulations 
    """
    from project.src.preprocessing.dataset_manager import DatasetManager
    return DatasetManager(config)



from testing_scripts.test_classification import test_classification


if __name__ == "__main__":
    """
    ***********************
    Setup
    ***********************
    """
    # 1. load configuration details
    config = load_config()
    # Set up a local Dask cluster
    dask_client = init_local_dask()
    
    """
    ***********************
    Global data preparation
    ***********************
    """
    # 1. get dataset manager to handle dataset manipulations
    dataset_manager = init_dataset_manager(config)

    # 2. convert raw data to parquet (raw -> interim) 
    # 3. get interim data statistics    
    # 4. store statistics
    
    # 5. prepare training and test data sets (interim -> preprocessed)
    dataset_manager.write_dataset(test_size=0.5, overwrite=False)
    
    
    """
    ***********************
    Local data preparation
    ***********************
    """
    partitions_sample = 10
    target = 'Status'
    import dataset.dataloader as dl
    dataloader = dl.DataLoader(dataset_manager)

    """
    ***********************
    Linear Model Experiment
    ***********************
    """
    # local preprocessing (e.g., OHE for linear models)    
    X_train, y_train = dataloader.get_train_ohe(cat_columns_to_ohe, date_columns_to_ohe)
    X, y = dl.convert_to_pandas(X_train, y_train, target, partitions_sample)
    X.fillna(0, inplace=True)

    """
    Fit models to training data and test against test data
    """
    logistic_reg_save = Path(config.models_directory, 'train_02_logistic_regression.sav')    
    from training_scripts.train_02_logistic_regression import train_logistic_regression
    linr_train = train_logistic_regression(X, y, save_to=logistic_reg_save, recompute=recompute)

    """
    Test performance on unseen data
    """    
    X_test, y_test = dataloader.get_test_ohe(cat_columns_to_ohe, date_columns_to_ohe)
    X, y = dl.convert_to_pandas(X_test, y_test, target, partitions_sample)
    X.fillna(0, inplace=True)
    test_classification(X, y, logistic_reg_save)
    
    """
    Get interpretation
    """
    import eli5
    # apply interpretability strategy
    explanation_df = eli5.explain_weights_df(joblib.load(logistic_reg_save), feature_names=X.columns.values)
    explanation_df.sort_values('weight', inplace=True)
    print(explanation_df.head())
    

    """
    ****************************
    Gradient Boosting Experiment
    ****************************
    """
    
    
#     X, y = lightgbm_preprocess(X_train, 
#                                 y_train,
#                                 target,
#                                 partitions_sample)
    
#     """
#     LightGBM parameters
#     """
#     params = {
#                 "max_bin": 512,
#                 "learning_rate": 0.05,
#                 "boosting_type": "gbdt",
#                 "objective": "binary",
#                 "metric": "binary_logloss",
#                 "num_leaves": 10, # Low value will decrease likelihood of overfitting
#                 "verbose": -1,
#                 "min_data": 100,
#                 "boost_from_average": True}
    
#     """
#     Training
#     """
    
#     lgb_reg_filename = 'lgb_ran_estimator.sav'
#     lgb_reg_save = Path(config.models_directory, lgb_reg_filename)

#     lgb_model = train_model_lightgbm(X, y, params, lgb_reg_save)
    
    
#     """
#     Test
#     """
    
#     X, y = lightgbm_preprocess(X_test, 
#                                 y_test,
#                                 target,
#                                 partitions_sample)
    
#     test_classification(X, y, lgb_model)

    



