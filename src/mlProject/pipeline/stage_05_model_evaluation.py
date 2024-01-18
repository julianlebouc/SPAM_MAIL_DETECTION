from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.model_evaluation import ModelEvaluation

from src.mlProject import logger
from pathlib import Path

from src.mlProject import logger

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        config=ConfigurationManager()
        model_evaluation_config=config.get_model_evaluation_config()
        model_trainer=ModelEvaluation(config=model_evaluation_config)
        model_trainer.log_into_mlflow()
