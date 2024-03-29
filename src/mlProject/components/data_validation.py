
import os, os.path

from src.mlProject import logger

from src.mlProject.entity.config_entity import DataValidationConfig

import pandas as pd


class DataValidation:
    def __init__(self,config: DataValidationConfig ):
        self.config = config

    def validation_all_columns(self)->bool:
        try:
            _, _, files = next(os.walk(self.config.unzip_data_dir)) 
            nb_of_files = len(files) 
            print("nb de fichiers : ", nb_of_files)
            right_file_name_format = True
            for file_name in files:
                if 'msg' not in file_name and 'spmsg' not in file_name:
                    right_file_name_format = False

            validation_status = nb_of_files and right_file_name_format
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f'Validation status : {validation_status}')
            return validation_status
        except Exception as e:
            raise e