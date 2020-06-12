from pathlib import Path
import joblib
recompute=True

# Columns to perform OHE on (applies only for linear case)
cat_columns_to_ohe = ['HourSlot', 'Service']
date_columns_to_ohe = []

# Testing modules
# from testing_scripts.test_regression import test_regression  
from testing_scripts.test_classification import test_classification


def load_config(config_path='config.ini'):
    """
    Loads common configuration parameters
    """
    from project.utils.setup.configuration_manager import Config
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
    from project.utils.preprocessing.dataset_manager import DatasetManager
    return DatasetManager(config)


def run_exp_02_logistic_regression():
    """
    ***********************
    Linear Model Experiment
    ***********************
    """
    binary_target = ()
    
    # local preprocessing (e.g., OHE for linear models)    
    X_train, y_train = dataloader.get_train_ohe(cat_columns_to_ohe, date_columns_to_ohe)
    X, y = dl.convert_to_pandas(X_train, y_train, target, partitions_sample)
    X.fillna(0, inplace=True)

    """
    Fit model to training data
    """
    model_path = Path(config.models_directory, 'train_02_logistic_regression.sav')    
    from training_scripts.train_02_logistic_regression import train_logistic_regression
    train_logistic_regression(X, y, save_to=model_path, recompute=recompute)

    """
    Test performance on unseen data
    """    
    X_test, y_test = dataloader.get_test_ohe(cat_columns_to_ohe, 
                                             date_columns_to_ohe)
    X, y = dl.convert_to_pandas(X_test, y_test, target, partitions_sample)
    X.fillna(0, inplace=True)
    test_classification(X, y, model_path)
    
    """
    Get interpretation - simple weights for linear models
    """
    import eli5
    explanation_df = eli5.explain_weights_df(joblib.load(model_path), 
                                             feature_names=X.columns.values)
    explanation_df.sort_values('weight', inplace=True)
    print(explanation_df.head())

    
def run_exp_04_lightgbm_classification():
    """
    ****************************
    Gradient Boosting Experiment
    ****************************
    """
    
    # local preprocessing (e.g., OHE for linear models)    
#     X_train, y_train = dataloader.get_train()
    X_train, y_train = dataloader.get_train_ohe(cat_columns_to_ohe, 
                                                date_columns_to_ohe,
                                                'Status',
                                                binary_target_tup = ('Normal', 'Blocked')
                                               )
    X, y = dl.convert_to_pandas(X_train, y_train, target, partitions_sample)
    
    """
    Fit model to training data
    """
    model_path = Path(config.models_directory, 'train_04_lightgbm_classification.sav')    

    from training_scripts.train_04_lightgbm_classification import train_lightgbm_regression
    train_lightgbm_regression(X, y, save_to=model_path, recompute=recompute)
    
    """
    Test performance on unseen data
    """    
#     X_test, y_test = dataloader.get_test()
    X_test, y_test = dataloader.get_test_ohe(cat_columns_to_ohe, 
                                             date_columns_to_ohe, 
                                             'Status',
                                             binary_target_tup = ('Normal', 'Blocked')
                                            )
    X, y = dl.convert_to_pandas(X_test, y_test, target, partitions_sample)
    
    test_classification(X, y, model_path)
    



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

    run_exp_02_logistic_regression()
    
#     run_exp_04_lightgbm_classification()



