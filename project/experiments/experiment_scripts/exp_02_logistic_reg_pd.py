import exp_setup as setup
from pathlib import Path

def run_exp_02_logistic_regression_pd(model_path, retrain_model=False):
    """
    Linear Model Experiment
    """
    
    # local preprocessing (e.g., OHE for linear models)    
    X, y = dataloader.get_train_ohe(dask=False)
    X.fillna(0, inplace=True)

    """
    Fit model to training data
    """
    from project.experiments.training_scripts.train_02_logistic_regression import train_logistic_regression
    train_logistic_regression(X, y, save_to=model_path, recompute=retrain_model)

    """
    Test performance on unseen data
    """    
    X, y = dataloader.get_test_ohe(dask=False)
    X.fillna(0, inplace=True)
    
    from project.experiments.testing_scripts.test_regression import test_classification
    test_classification(X, y, model_path)
    
    """
    Get interpretation - simple weights for linear models
    """
    import eli5
    explanation_df = eli5.explain_weights_df(joblib.load(model_path), 
                                             feature_names=X.columns.values)
    explanation_df.sort_values('weight', inplace=True)
    print(explanation_df.head())


    
if __name__ == "__main__":

    # Load config
    config_path = '../config.ini'
    config = setup.load_config(config_path)
    
    # Start dask client
    dask_client = setup.init_local_dask()
    
    # Create dataset manager
    dataset_manager = setup.init_dataset_manager(config)
    dataset_manager.write_dataset(test_size=0.5, overwrite=False)
    
    
    # Get dataloader
    dataloader = setup.init_dataloader(dataset_manager)

    # Run experiment
    model_name = 'exp_2_log_reg.sav'
    model_path = Path('../', config.models_directory, model_name) 
    retrain_model=True

    run_exp_02_logistic_regression_pd(model_path, retrain_model)
    

    