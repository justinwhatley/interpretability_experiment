from configparser import ConfigParser
from os.path import isfile, join

class Config():
    """
    Loading config
    """
    def __init__(self, config_path):
        self.parser = ConfigParser()
        if isfile(config_path):
            print('Loading configuration from: ' + str(config_path))
            self.parser.read(config_path)
            self.set_file_details()

        else:
            print('Config file not found: ' + str(config_path))
        
    def set_file_details(self):
        self.input_path = join(self.parser['file_location']['data_directory'], 
                                    self.parser['file_location']['input_data_dir'])
        
        self.output_directory = self.parser['results']['output_directory']



        
        