import exp_setup
from pathlib import Path

def run_exp_4_lightgbm_classification_pd(model_path, retrain_model=False):
    """
    Linear Model Experiment
    """
    
    # local preprocessing (e.g., OHE for linear models)    
    X, y = dataloader.get_train_ohe(dask=False)
    X.fillna(0, inplace=True)

    """
    Fit model to training data
    """
    from project.experiments.training_scripts.train_04_lightgbm_classification import train_lightgbm_classification
    train_lightgbm_classification(X, y, save_to=model_path, recompute=retrain_model)

    """
    Test performance on unseen data
    """    
    X, y = dataloader.get_test_ohe(dask=False)
    X.fillna(0, inplace=True)
    
    from project.experiments.testing_scripts.test_classification import test_classification
    test_classification(X, y, model_path)


    
if __name__ == "__main__":

    config, dask_client, dataset_manager, dataloader = exp_setup.run_setup()
    retrain_model = True

    """
    ***************************
    Run experiment (edit below)
    ***************************
    """

    model_name = 'exp_4_lgbm_class.sav'
    model_path = Path('../', config.models_directory, model_name) 
    run_exp_4_lightgbm_classification_pd(model_path, retrain_model)
    

    