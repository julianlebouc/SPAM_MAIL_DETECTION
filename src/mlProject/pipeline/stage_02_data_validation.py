from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_validation import DataValidation
from mlProject import logger


from mlProject import logger

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
    
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation=DataValidation(config=data_validation_config)
        data_validation.validation_all_columns()
