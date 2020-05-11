from torch.utils.data import Dataset
import numpy as np

"""
Separates dataset 
"""
class PrepareDataset(Dataset):
    def __init__(self, X, y, embedded_column_names):
        
        # Isolate categorical columns
        self.X1 = X.loc[:,embedded_column_names].copy().values.astype(np.int64) 
#         print(self.X1)
        
        # Isolate numerical columns
        self.X2 = X.drop(columns = embedded_column_names).copy().values.astype(np.float64) 
#         print(self.X2)

        self.y = y
#         print(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]