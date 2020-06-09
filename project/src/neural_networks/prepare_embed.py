from torch.utils.data import Dataset
import numpy as np
import torch

"""
Separates dataset 
"""
class PrepareDataset(Dataset):
    def __init__(self, X, y, embedded_column_names):
        # Isolate categorical columns
        self.X1 = X.loc[:,embedded_column_names].copy().values.astype(np.int64) 

        # Isolate numerical columns
        self.X2 = X.drop(columns = embedded_column_names).copy().values.astype(np.float64)         
        self.y = y.values.astype(np.float64) 

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]