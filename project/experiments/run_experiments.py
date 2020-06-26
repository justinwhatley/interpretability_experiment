import experiment_scripts.exp_setup as exp_setup
from pathlib import Path

# Testing modules


    
def run_exp_05_lightgbm_classification_incr(model_sav):
    """
    ****************************************
    Incremental Gradient Boosting Experiment
    ****************************************
    """
    
    # local preprocessing (e.g., OHE for linear models)    
#     X_train, y_train = dataloader.get_train()
    X, y = dataloader.get_train_ohe(cat_columns_to_ohe, date_columns_to_ohe, dask=True)
    
    """
    Fit model to training data
    """
    model_path = Path(config.models_directory, 'train_05_lightgbm_classification.sav')    

    from training_scripts.train_04_lightgbm_classification import train_lightgbm_regression
    train_lightgbm_regression(X, y, save_to=model_path, recompute=recompute)
    
    """
    Test performance on unseen data
    """    
#     X_test, y_test = dataloader.get_test()
    X, y = dataloader.get_test_ohe(cat_columns_to_ohe, date_columns_to_ohe, dask=True)
    
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
    
    target = 'Status'
    import dataset.dataloader as dl
    dataloader = dl.DataLoader(dataset_manager)

    run_exp_02_logistic_regression_pd()
    run_exp_04_lightgbm_classification_pd()
#     run_exp_05_lightgbm_classification('train_05_lightgbm_classification.sav')



