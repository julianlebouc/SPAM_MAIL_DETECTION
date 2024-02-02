import pandas as pd
import os
import pickle
from src.mlProject import logger
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

import joblib


from src.mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config=config
    

    def train(self):
        with open(self.config.train_data_path, 'rb') as file:
            data = pickle.load(file)
        with open(self.config.test_data_path, 'rb') as file:
            classes = pickle.load(file)
        
        X_train, X_test, y_train, y_test = train_test_split(data, classes, stratify=classes, test_size=0.2)
        clf = AdaBoostClassifier(n_estimators=self.config.n_estimators, learning_rate=self.config.learning_rate,algorithm=self.config.algorithm)
        clf.fit(X_train, y_train)

        joblib.dump(clf,os.path.join(self.config.root_dir,self.config.model_name))
