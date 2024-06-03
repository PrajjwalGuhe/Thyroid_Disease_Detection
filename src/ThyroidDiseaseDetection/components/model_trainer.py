import numpy as np
import os
import sys
import pandas as pd
from sklearn.impute import SimpleImputer  #handling missing values
from sklearn.preprocessing import StandardScaler # handling feature scaling
from sklearn.preprocessing import OrdinalEncoder #it will assign 1,2,3,4 to our categorial 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from src.ThyroidDiseaseDetection.logger import logging
from src.ThyroidDiseaseDetection.utils.utils import save_object
from src.ThyroidDiseaseDetection.exception import customexception
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Spitting dependent and indepedent variable')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            param_dist = {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 10),
                'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                'subsample': [0.5, 0.7, 0.9, 1.0],
                'max_features': ['auto', 'sqrt', 'log2'],
                'loss': ['ls', 'lad', 'huber', 'quantile']
            }

            # Initialize Gradient Boosting Regressor
            gb = GradientBoostingRegressor()

            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(estimator=gb, param_distributions=param_dist,
                                            n_iter=100, cv=5, scoring='neg_mean_squared_error',
                                            random_state=42, n_jobs=-1)

            # Fit the RandomizedSearchCV
            random_search.fit(X_train, y_train)

            # Best hyperparameters
            print("Best hyperparameters:", random_search.best_params_)

            # Evaluate performance
            best_model = random_search.best_estimator_
            mse = mean_squared_error(y_test, best_model.predict(X_test))
            logging.info(f"MSE on test set:, {mse}")

            hyperparameters = random_search.best_params_

            logging.info(f'best hyperparameter found in gradient boosting {hyperparameters}')

            model = GradientBoostingRegressor(**hyperparameters)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            logging.info(f'Accuracy of the model: {r2_score(y_test, y_pred)}')


            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
        
        except Exception as e:
            logging.info('Exception occured at model training')
            raise customexception(e, sys)