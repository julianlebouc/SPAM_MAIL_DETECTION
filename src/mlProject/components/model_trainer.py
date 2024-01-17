import pandas as pd
import os
import pickle
from mlProject import logger
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

import joblib


from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config=config
    

    def train(self):
        with open(self.config.train_data_path, 'rb') as file:
            data = pickle.load(file)
        classes = pd.read_csv(self.config.test_data_path)
        
        X_train, X_test, y_train, y_test = train_test_split(data, classes, stratify=classes, test_size=0.2)
        clf = AdaBoostClassifier()
        clf.fit(X_train, y_train)

        joblib.dump(clf,os.path.join(self.config.root_dir,self.config.model_name))
