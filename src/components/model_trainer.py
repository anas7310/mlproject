import os
import sys
from dataclasses import dataclass
from src.logger import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator

from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class CompatibleCatBoostRegressor(CatBoostRegressor, BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data.')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CompatibleCatBoostRegressor(verbose=False, allow_writing_files=False),
                "GradientBoosting":GradientBoostingRegressor()
            }

            params = {
                "Linear Regression": {},
                "Ridge": {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                },
                "Lasso": {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'selection': ['cyclic', 'random']
                },
                "SVR": {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.1, 1, 10, 100],
                    'epsilon': [0.1, 0.2, 0.5]
                },
                "KNN": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [None, 5, 10]
                },
                "AdaBoost": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'loss': ['linear', 'square', 'exponential']
                },
                "XGBoost": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0]
                },
                "CatBoost": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                    'verbose': [False]  # Keeps the output clean during tuning
                },
                "GradientBoosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')

            logging.info(f'Best model found is: {best_model_name}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            test_score = r2_score(y_test, predicted)
            return test_score

        except Exception as e:
            raise CustomException(e,sys)