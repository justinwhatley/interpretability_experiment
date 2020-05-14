import dask.dataframe as dd
import fastparquet
from pathlib import PurePath

class DatasetBuilder():
    """
    Used to create a training and test set to be used by different scripts
    """
    
    def __init__(self, config_obj, test_size=0.1):
        self.config = config_obj
        
        self.set_directories(self.config)
        self.set_targets()
        
        self.ddf = self.load_ddf()
        self.ddf = self.make_modifications(self.ddf, self.config)
        
        self.prepare_training_test(test_size)
    
    
    def set_directories(self, config):
        self.input_path = config.input_path
        
        # Preprocessed directory
        self.train_data_path = PurePath(config.train_data)
        self.train_target_path = PurePath(config.train_target)

        self.test_data_path = PurePath(config.test_data)
        self.test_target_path = PurePath(config.test_target)    
    
    
    def load_ddf(self):
        return dd.read_parquet(self.input_path)
        

    def set_targets(self):
        self.targets = self.config.targets
        
    
    def make_modifications(self, ddf, config):
        """
        Modifies the columns before output to test and training sets
        
        mod_dict_keys = 'new_column_name', 'first_column', 'operation', 'second_column'
        """
        mods = config.to_modify
        for mod in mods:
            if mod['operation'].lower() == 'subtract':
                ddf[mod['new_column_name']] = ddf[mod['first_column']] - ddf[mod['second_column']]
            
            elif mod['operation'].lower() == 'divide':
                ddf[mod['new_column_name']] = ddf[mod['first_column']] / ddf[mod['second_column']]
                
            else:
                print('Operation is not yet supported')
        
        return ddf
        
        
    def prepare_training_test(self, test_size = 0.1):
        """
        Separates to training a test sets (once dask is run)
        """
        from dask_ml.model_selection import train_test_split
    
        y = self.ddf[self.targets]
        X = self.ddf.drop(self.targets, axis=1)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, 
                                                                                y, 
                                                                                test_size = test_size,
                                                                                random_state = 42)

        
    def write_dataset(self, overwrite = False):
        """
        Write dataset to directory
        """
        import pyarrow
        engine = 'pyarrow'  
        # TODO handle overwriting 
        dd.to_parquet(self.X_train, self.train_data_path, engine=engine)
        dd.to_parquet(self.y_train, self.train_target_path, engine=engine)

        dd.to_parquet(self.X_test, self.test_data_path , engine=engine)
        dd.to_parquet(self.y_test, self.test_target_path, engine=engine)
    
      
    