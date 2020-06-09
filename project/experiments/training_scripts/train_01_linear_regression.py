   
    
    
    
    
def

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