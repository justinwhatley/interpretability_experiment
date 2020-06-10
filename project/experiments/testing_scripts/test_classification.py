import joblib

# class ClassificationTester():
#     """
#     Test regression performance
#     """
#     def __init__(self, dataloader):

#         self.dataloader = dataloader
        


def test_classification(X, y, model_path):
    """
    In memory testing of linear regression, loading model from path
    """
    model = joblib.load(model_path)
    
    from sklearn.metrics import classification_report

    predictions = model.predict(X)
    print(classification_report(predictions, y))