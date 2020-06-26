import exp_setup
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
    
    from project.experiments.testing_scripts.test_classification import test_classification
    test_classification(X, y, model_path)
    
    """
    Get interpretation - simple weights for linear models
    """
    import eli5
    import joblib
    explanation_df = eli5.explain_weights_df(joblib.load(model_path), 
                                             feature_names=X.columns.values)
    explanation_df.sort_values('weight', inplace=True)
    print(explanation_df.head())


    
if __name__ == "__main__":

    config, dask_client, dataset_manager, dataloader = exp_setup.run_setup()
    retrain_model = True

    """
    ***************************
    Run experiment (edit below)
    ***************************
    """

    model_name = 'exp_2_lr_class.sav'
    model_path = Path('../', config.models_directory, model_name) 
    run_exp_02_logistic_regression_pd(model_path, retrain_model)
    

    