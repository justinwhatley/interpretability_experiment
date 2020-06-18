

# from pathlib import Path

# filename = Path(__file__).stem
# print('Running: ' + str(filename))
    
# model_filename = Path(filename, '.sav')
# logistic_reg_save = Path(config.models_directory, logr_reg_filename)    

from project.utils.models.model_handling import load_or_store_model
@load_or_store_model
def train_logistic_regression(X, y, save_to, recompute=False):
    """
    In memory training of linear regression
    """
    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42)
    estimator.fit(X, y=y)

    return estimator



