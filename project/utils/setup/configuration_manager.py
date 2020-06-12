from configparser import ConfigParser
from os.path import isfile, join
from os import chdir
from os import getcwd

class Config():
    """
    Loading config
    """
    def __init__(self, config_path, verbose=False):
        self.parser = ConfigParser()
        self.parser.optionxform = str
                
        if isfile(config_path):
            print('Loading configuration from: ' + str(config_path))
            self.parser.read(config_path)
            self.get_file_details(verbose)
            self.get_modifications()
            self.get_targets()

        else:
            print('Config file not found: ' + str(config_path))
        
        
    def get_file_details(self, verbose):
        
        directories = 'directories'

        self.raw_data_path = join(self.parser[directories]['raw_data_path'])
        
        
        data_directory = join(self.parser[directories]['base_directory'],
                              self.parser[directories]['data_directory'])
                              
        
        # Interim data directory (may include fewer columns or preliminary preprocessing from raw)
        self.input_path = join(data_directory, 
                               self.parser[directories]['interim_data_dir'],
                               self.parser[directories]['interim_data'])

        
        # Preprocessed data directory
        self.preprocessed_dir = join(data_directory,
                                     self.parser[directories]['preprocessed_data_dir'])
        
        self.train_data = join(self.preprocessed_dir, 
                                    self.parser[directories]['train_data'])
        
        self.test_data = join(self.preprocessed_dir, 
                                    self.parser[directories]['test_data'])
        
        
        # Figures directory
        self.figures_dir = self.parser['results']['figures']
        
        # Trained models directory
        self.models_directory = self.parser['results']['trained_models']
        
        if verbose:
            print('raw_input: ' + self.raw_data_path)
            print('input_path: ' + self.input_path)
            print('figures_dir: ' + self.figures_dir)
            print('preprocessed_dir: ' + self.preprocessed_dir)
            print('train_data: ' + self.train_data)
            print('train_target: ' + self.train_target)
            print('test_data: ' + self.test_data)
            print('test_target: ' + self.test_target)


    def get_inputs(self):
        pass
        
            
    def get_targets(self):
        target_column = 'target_column' 
        self.target_column = self.parser[target_column]['target'].strip().split()[0]
        
        self.target_categories = self.parser[target_column]['target_categories'].strip().split()


    def get_modifications(self):
        modified_columns = 'modified_columns'
        
        # Get columns to add from the configuration. This includes operations between columns
        to_modify_lst = self.parser[modified_columns]['add_columns'].strip().split('\n')

        self.to_modify = []
        for modification_line in to_modify_lst:
            r_lst = modification_line.split(',')
            if len(r_lst) == 4: 
                r_dict =  {'new_column_name': r_lst[0].strip(), 
                           'first_column': r_lst[1].strip(), 
                           'operation': r_lst[2].strip(),
                           'second_column': r_lst[3].strip()}
            elif len(r_lst) == 3:
                r_dict =  {'new_column_name': r_lst[0].strip(), 
                           'first_column': r_lst[1].strip(), 
                           'operation': r_lst[2].strip()}
                
            try:
                # Handles case where r_dict not available
                self.to_modify.append(r_dict)
            except:
                pass
    
        # Get columns to drop from the training/validation set
        self.columns_to_drop = self.parser[modified_columns]['drop_columns'].strip().split('\n')

