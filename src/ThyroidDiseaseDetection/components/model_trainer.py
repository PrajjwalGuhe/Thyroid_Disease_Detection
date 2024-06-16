import numpy as np
import os
import sys
import pandas as pd
from sklearn.impute import SimpleImputer  # handling missing values
from sklearn.preprocessing import StandardScaler, OrdinalEncoder  # handling feature scaling and encoding
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from src.ThyroidDiseaseDetection.logger import logging
from src.ThyroidDiseaseDetection.utils.utils import save_object
from src.ThyroidDiseaseDetection.exception import customexception
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting dependent and independent variables')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression': {
                    'model': LinearRegression(),
                    'params': {
                        'fit_intercept': [True, False],
                        'copy_X': [True, False],
                        'positive': [True, False]
                    }
                },
                'Ridge Regression': {
                    'model': Ridge(),
                    'params': {
                        'alpha': uniform(0.1, 10),
                        'fit_intercept': [True, False]
                    }
                },
                'Lasso Regression': {
                    'model': Lasso(),
                    'params': {
                        'alpha': uniform(0.1, 10),
                        'fit_intercept': [True, False]
                    }
                },
                'ElasticNet Regression': {
                    'model': ElasticNet(),
                    'params': {
                        'alpha': uniform(0.1, 10),
                        'l1_ratio': uniform(0, 1),
                        'fit_intercept': [True, False]
                    }
                },
                'Random Forest': {
                    'model': RandomForestRegressor(),
                    'params': {
                        'n_estimators': randint(50, 500),
                        'max_features': ['sqrt', 'log2', None],
                        'max_depth': randint(3, 10)
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingRegressor(),
                    'params': {
                        'n_estimators': randint(50, 500),
                        'max_depth': randint(3, 10),
                        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                        'subsample': [0.5, 0.7, 0.9, 1.0],
                        'max_features': ['sqrt', 'log2', None],
                        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile']
                    }
                },
                'AdaBoost': {
                    'model': AdaBoostRegressor(),
                    'params': {
                        'n_estimators': randint(50, 500),
                        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3]
                    }
                },
                'Support Vector Regressor': {
                    'model': SVR(),
                    'params': {
                        'C': uniform(0.1, 10),
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'epsilon': uniform(0.01, 1)
                    }
                },
                'Decision Tree': {
                    'model': DecisionTreeRegressor(),
                    'params': {
                        'max_depth': randint(3, 10),
                        'min_samples_split': randint(2, 20),
                        'min_samples_leaf': randint(1, 20)
                    }
                },
                'Extra Tree': {
                    'model': ExtraTreeRegressor(),
                    'params': {
                        'max_depth': randint(3, 10),
                        'min_samples_split': randint(2, 20),
                        'min_samples_leaf': randint(1, 20)
                    }
                },
                'K-Neighbors Regressor': {
                    'model': KNeighborsRegressor(),
                    'params': {
                        'n_neighbors': randint(1, 30),
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan', 'minkowski']
                    }
                }
            }

            # Function to evaluate models
            def evaluate_models(models, X_train, X_test, y_train, y_test):
                results = {}
                best_model = None
                best_score = float('-inf')
                best_params = None

                for model_name, model_info in models.items():
                    model = model_info['model']
                    params = model_info['params']
                    
                    random_search = RandomizedSearchCV(estimator=model, param_distributions=params,
                                                       n_iter=100, cv=5, scoring='neg_mean_squared_error',
                                                       random_state=42, n_jobs=-1)
                    random_search.fit(X_train, y_train)

                    best_estimator = random_search.best_estimator_
                    y_pred = best_estimator.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    results[model_name] = {
                        'Best Estimator': best_estimator,
                        'Best Params': random_search.best_params_,
                        'MAE': mae,
                        'MSE': mse,
                        'RMSE': rmse,
                        'R²': r2
                    }

                    if r2 > best_score:
                        best_model = model_name
                        best_score = r2
                        best_params = random_search.best_params_

                return results, best_model, best_score, best_params

            # Evaluate the models
            results, best_model, best_score, best_params = evaluate_models(models, X_train, X_test, y_train, y_test)

            # Print the results
            for model_name, metrics in results.items():
                print(f'\nModel: {model_name}')
                print(f'Best Params: {metrics["Best Params"]}')
                print(f'MAE: {metrics["MAE"]}')
                print(f'MSE: {metrics["MSE"]}')
                print(f'RMSE: {metrics["RMSE"]}')
                print(f'R²: {metrics["R²"]}')

            print(f'\nBest Model: {best_model}')
            print(f'Best Score (R²): {best_score}')
            print(f'Best Parameters: {best_params}')

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=results[best_model]['Best Estimator']
            )

        except Exception as e:
            logging.info('Exception occurred during model training')
            raise customexception(e, sys)
