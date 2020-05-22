# def normalize_df_cont(df, continuous_columns):
#     """
#     Normalize the continuous colums in a dataframe  
#     """
#     means, stds = {},{}
#     for column in continuous_columns:
#         means[column], stds[column] = df[column].mean(), df[column].std()
#         df[column] = (df[column]-means[column]) / (1e-7 + stds[column])
"""
Inspired by:
https://nbviewer.jupyter.org/github/PuneetGrov3r/MediumPosts/blob/master/Tackle/BigData-IncrementalLearningAndDask.ipynb#Method-2:-Using-Dask:
"""

import dask
from dask_ml.preprocessing import RobustScaler

class DDFRescaler():
    """
    Uses robust scaler which scales the data according to the quantile range
    Chunks the data to prevent prodcasting errors given different sized ddf partitions
    """
    def __init__(self, X_ddf, y_ddf, chunk_length = 200000):
        
        self.lengths = self.get_partitions_lengths(y_ddf)
        self.chunk_length = chunk_length
        
        self.y = self.rescale_target(y_ddf)
        self.X = self.rescale_features(X_ddf)
        

    def rescale_target(self, y_ddf):
        # Conversion from Dask series to Dask array
        y = y_ddf.to_dask_array(lengths=self.lengths)
        
        rsc = RobustScaler()
        y = rsc.fit_transform(y.reshape(-1, 1)).reshape(1, -1)[0]
        return y
        
        
    def rescale_features(self, X_ddf):
        # Conversion from Dask dataframe to Dask array
        X = X_ddf.to_dask_array(lengths=self.lengths)
        
        # Resizing blocks in order to prevent broadcasting errors due to different input sizes
        Xo = dask.array.zeros((X.shape[0],1), chunks=(self.chunk_length,1))
        
        for i, _col in enumerate(X_ddf.columns):
            rsc = RobustScaler()
            temp = rsc.fit_transform(X[:,i].reshape(-1, 1))
            Xo = dask.array.concatenate([Xo, temp], axis=1)
            
        return Xo
        
    def get_partitions_lengths(self, ddf):
        """
        Gets the length of each partitions and saves/returns it
        """
        lengths = []
        for part in ddf.partitions:
            l = part.shape[0].compute()
            lengths.append(l)
        return lengths
        
        
        
    

    
    
#     for i, col_ in enumerate(ddf[input_columns + [target]].columns):
#         if col_ == target:
            
#         else:
#             rsc = RobustScaler()
#             temp = rsc.fit_transform(X[:,i].reshape(-1, 1))
#             Xo = dask.array.concatenate([Xo, temp], axis=1)