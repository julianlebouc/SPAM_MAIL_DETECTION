from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.data_validation import DataValidation
from src.mlProject import logger


from src.mlProject import logger

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
    
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation=DataValidation(config=data_validation_config)
        data_validation.validation_all_columns()
