import joblib
from sklearn.metrics import classification_report


def test_classification(X, y, model_path):
    """
    In memory testing of linear regression, loading model from path
    Note: Assumes binary classification between 0-1
    """
    model = joblib.load(model_path)
    
    from sklearn.metrics import classification_report

    predictions = model.predict(X)

    # Rounds the prediction assuming binary targets 0-1
    print('Classification results: ')
    print(classification_report(y, predictions.round()))
    
    
def test_classification_ddf(X_ddf, y_ddf, model_path):
    """
    In memory testing of linear regression, loading model from path
    Note: Assumes binary classification between 0-1
    """
    model = joblib.load(model_path)

    
    with joblib.parallel_backend('dask'):
    
        predictions = model.predict(X)

        # Rounds the prediction assuming binary targets 0-1
        print('Classification results: ')
        print(classification_report(y, predictions.round()))

    