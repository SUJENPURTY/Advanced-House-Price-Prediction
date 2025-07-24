# Basic Imports
import numpy as np # type: ignore
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import save_object, evaluate_regression_model  # Use regression evaluator

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting dependent and independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Regression Models
            models = {
                'LinearRegression': LinearRegression(),
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'SVR': SVR()
            }

            model_report = evaluate_regression_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            logging.info(f'Model Report: {model_report}')
            print('\n' + '='*80 + '\n')

            # Select the best model (by highest R2 score)
            best_model_name = None
            best_model_score = float("-inf")

            for model_name, scores in model_report.items():
                if scores["R2 Score"] > best_model_score:
                    best_model_score = scores["R2 Score"]
                    best_model_name = model_name

            best_model = models[best_model_name] # type: ignore

            print(f'Best Model Found: {best_model_name} with R2 Score: {best_model_score}')
            logging.info(f'Best Model Found: {best_model_name} with R2 Score: {best_model_score}')
            print('\n' + '='*80 + '\n')

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.error('Exception occurred during Model Training')
            raise CustomException(e, sys)  # type: ignore
