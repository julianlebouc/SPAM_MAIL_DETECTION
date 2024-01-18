import os
import pandas as pd
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score



from src.mlProject.entity.config_entity import ModelEvaluationConfig
from src.mlProject.constants import *
from src.mlProject.utils.common import read_yaml, create_directories,save_json



class ModelEvaluation:
    def __init__(self,config : ModelEvaluationConfig):
        self.config=config

    def eval_metrics(self, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred,  pos_label="spam")
        prec = precision_score(y_test, y_pred,  pos_label="spam")
        return acc, recall, prec
    
    def log_into_mlflow(self):
        with open(self.config.train_data_path, 'rb') as file:
            data = pickle.load(file)
        with open(self.config.test_data_path, 'rb') as file:
            classes = pickle.load(file)
        
        X_train, test_x, y_train, test_y = train_test_split(data, classes, stratify=classes, test_size=0.2)
        model=joblib.load(self.config.model_path)


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            predicted_qualities= model.predict(test_x)
            (acc, recall, prec)= self.eval_metrics(test_y,predicted_qualities)
            #Saving metrics as local
            scores={"acc":acc,"recall":recall,"prec":prec}
            save_json(path=Path(self.config.metric_file_name),data=scores)
            
            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("acc",acc)
            mlflow.log_metric("prec",prec)
            mlflow.log_metric("recall",recall)
            
            # Model registry does not work with file store
            if tracking_url_type_store !="file":

                mlflow.sklearn.log_model(model,"model",registered_model_name="AdaBoost")
            else:
                mlflow.sklearn.log_model(model,"model")

