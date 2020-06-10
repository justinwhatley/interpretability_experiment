import joblib

class RegressionTester():
    """
    Test a regression 
    """
    def __init__(self, dataloader, model_path):
        self.dataloader = dataloader
        self.model = joblib.load(model_path)
        
        
    def test_regression():
        pass
    
