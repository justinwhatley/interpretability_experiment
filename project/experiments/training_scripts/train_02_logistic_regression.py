

# from pathlib import Path

# filename = Path(__file__).stem
# print('Running: ' + str(filename))
    
# model_filename = Path(filename, '.sav')
# logistic_reg_save = Path(config.models_directory, logr_reg_filename)    
import joblib

def train_logistic_regression(X, y, save_to, recompute=False):
    """
    In memory training of linear regression
    """
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



