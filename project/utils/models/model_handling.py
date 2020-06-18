import joblib

def load_or_store_model(func):
    """
    Wrapper/decorator to check whether the model is already saved to return saved model instead of new training
    Function must have a 'save_to' filepath and 'recompute' bool must be defined
    """
    
    def loading_wrapper(*args, **kwargs):
        recompute = kwargs['recompute']
        save_to = kwargs['save_to']
        
        if not recompute:
            try:
                print('Loading previously trained model: ' + str(save_to))
                return joblib.load(save_to)
            except:
                print('Model not found: ' + str(save_to))
        
        print('Training: ' + func.__module__)
        model = func(*args, **kwargs)
        return save_model(model, save_to)

    def save_model(model, save_to):
        print('Saving model to: ' + str(save_to))
        joblib.dump(model, save_to)
        return model

    return loading_wrapper 




