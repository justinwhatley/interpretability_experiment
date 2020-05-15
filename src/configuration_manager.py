from configparser import ConfigParser
from os.path import isfile, join

class Config():
    """
    Loading config
    """
    def __init__(self, config_path):
        self.parser = ConfigParser()
        self.parser.optionxform = str
        
        if isfile(config_path):
            print('Loading configuration from: ' + str(config_path))
            self.parser.read(config_path)
            self.get_file_details()
            self.get_modifications()
            self.get_targets()

        else:
            print('Config file not found: ' + str(config_path))
        
        
    def get_file_details(self):
        
        file_location = 'file_location'
        self.input_path = join(self.parser[file_location]['data_directory'], 
                                    self.parser[file_location]['input_data_dir'])
        
        self.output_directory = self.parser['results']['output_directory']
        
        
        self.train_data_dir = join(self.parser[file_location]['data_directory'], 
                                    self.parser[file_location]['train_data_dir'])
        
        self.train_data = join(self.train_data_dir, 
                                    self.parser[file_location]['train_data'])
        
        self.train_target = join(self.train_data_dir, 
                                    self.parser[file_location]['train_target'])

        self.test_data_dir = join(self.parser[file_location]['data_directory'], 
                                    self.parser[file_location]['test_data_dir'])
        
        self.test_data = join(self.train_data_dir, 
                                    self.parser[file_location]['test_data'])
        
        self.test_target = join(self.train_data_dir, 
                                    self.parser[file_location]['test_target'])


    def get_targets(self):
        target_columns = 'target_columns' 
        self.targets = self.parser[target_columns]['targets'].strip().split()
 

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
                
            self.to_modify.append(r_dict)
    
        # Get columns to drop from the training/validation set
        self.columns_to_drop = self.parser[modified_columns]['drop_columns'].strip().split()

