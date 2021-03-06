import dask.dataframe as dd
import numpy as np
import fastparquet
from pathlib import PurePath
from os.path import isdir
from os import rmdir


class DatasetManager():
    """
    Separates files into training and test set to ensure that the test set data is excluded
    as a final test of the model
    """
    
    def __init__(self, config_obj):
        self.config = config_obj
        self.set_directories(self.config)
        self.ddf = None
        self.set_ohe_columns()
        self.set_targets()
    
    def set_directories(self, config):
        self.input_path = config.input_path
        
        # Preprocessed directory
        self.train_data_path = PurePath(config.train_data)

        self.test_data_path = PurePath(config.test_data)
    
    
    def load_ddf(self):
        if self.ddf is None:
            ddf = dd.read_parquet(self.input_path)
            self.ddf = self.make_modifications(ddf, self.config)
        return self.ddf

    
    def set_targets(self):
        self.target_column = self.config.target_column
        print(self.target_column)
        
        self.target_categories = None
        try:
            self.target_categories = self.config.target_categories
            print('Target categories are: ' + str(self.target_categories))
        except:
            pass
    
    
    def set_ohe_columns(self):
        self.cat_columns_to_ohe = self.config.cat_columns_to_ohe
        self.date_columns_to_ohe = self.config.date_columns_to_ohe
    
    
    def make_modifications(self, ddf, config):
        """
        Modifies the columns before output to test and training sets
        
        mod_dict_keys = 'new_column_name', 'first_column', 'operation', 'second_column'
        """
        mods = config.to_modify
        
        # Apply specified mod operations
        for mod in mods:
            if mod['operation'].lower() == 'subtract':
                ddf[mod['new_column_name']] = ddf[mod['first_column']] - ddf[mod['second_column']]
            
            elif mod['operation'].lower() == 'divide':
                ddf[mod['new_column_name']] = ddf[mod['first_column']] / ddf[mod['second_column']]
                # Sets all inf values to NA 
                ddf[mod['new_column_name']] = ddf[mod['new_column_name']].replace([np.inf, -np.inf], np.nan)
                # Records all inf, -inf and null as 0 - will need to consider whether to change this
                ddf[mod['new_column_name']] = ddf[mod['new_column_name']].fillna(0)
                
            elif mod['operation'].lower() == 'boolean':
                ddf[mod['new_column_name']] = ddf[mod['first_column']] > 0 
                
            else:
                print('Operation is not yet supported')
                
        # Drop specified columns
        for column in config.columns_to_drop:
            ddf = ddf.drop(column, axis=1)
        
        return ddf
        
       
    def _remove_dirs(self, paths_list):
        """
        Helper function to remove directory
        """
        print('Removing any old directories')
        for path in paths_list:
            try:
                rmdir(path)
            except OSError as e:
                pass
            

    def _check_all_paths_exist(self, paths_list):
        """
        Helper function to check whether all path already exist
        This ensures dataset creation is not skipped when the files are not already there
        """
        for path in paths_list:
            if not isdir(path):
                return False
        return True
      

    def write_dataset(self, test_size = 0.1, overwrite = False):
        """
        Write dataset to directory
        """
        
        paths_list = [self.train_data_path, 
                      self.test_data_path]
        
        all_paths_exist = self._check_all_paths_exist(paths_list)
        
        
        if overwrite or not all_paths_exist: 
            # Removes any full or partial datasets
            self._remove_dirs(paths_list)
            
            # Makes modifications from configuration file
            self.set_targets()
            self.ddf = self.load_ddf()
            
            # Writes training and test sets to disk
            self.separate_train_test(test_size)
            self._write_dataset()
        
        else:
            print('Not overwriting existing training and test sets')

            
    def _write_dataset(self):
        """
        Helper function to write the actual dataset files
        I've had better luck writing the files with pyarrow, reading with fastparquet
        """
        import pyarrow
        engine = 'pyarrow'  
        print('Writing trainings and test sets: ')
        
        dd.to_parquet(self.train_ddf, self.train_data_path, engine=engine)
#         dd.to_parquet(self.y_train, self.train_target_path, engine=engine)

        dd.to_parquet(self.test_ddf, self.test_data_path , engine=engine)
#         dd.to_parquet(self.y_test, self.test_target_path, engine=engine)
        
    
    
    def separate_train_test(self, test_fraction = 0.1):
        """
        Separates to training a test sets (once dask is run)
        """
#         from dask_ml.model_selection import train_test_split
    
#         y = self.ddf[self.targets]
#         X = self.ddf.drop(self.targets, axis=1)
        
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, 
#                                                                                 y, 
#                                                                                 test_size = test_size,
#                                                                                 random_state = 42)

        
#         from dask import random_split
        
        self.train_ddf, self.test_ddf = self.ddf.random_split([1-test_fraction, test_fraction] , random_state=42)

        
    def get_training_set(self):
        """
        Get dask training set reader
        """
        self.training_data = dd.read_parquet(self.train_data_path)
        
        return self.training_data
        
        
    def get_test_set(self):
        """
        Get dask test set reader
        """
        self.testing_data = dd.read_parquet(self.test_data_path)
        
        return self.testing_data
    
    
