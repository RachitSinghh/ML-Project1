import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl") # path to save the trained model

class ModelTrainer:
    def __init__(self): # initialize the model trainer
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array): # initiate the model trainer
        try:
            logging.info("Splitting training and test input data") # split the data into training and test sets

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifer": KNeighborsRegressor(),
                "XGBClassifer":XGBRegressor(), 
                "CatBoosting Classifer": CatBoostRegressor(verbose=False),
                "Adaboost Classifer": AdaBoostRegressor(),
            }    

            # params = {
            #     "Random Forest":{
            #         'n_estimators': [8,16,32,64,128,256], 
            #         "max_depth": [None, 8, 16, 32, 64, 128],
            #         "min_samples_split": [2, 5, 10],
            #     },
            #     "Decision Tree":{
            #         'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            #         "max_depth": [None, 5,10,20],
            #         "min_samples_split": [2, 5, 10],
            #     },
            #     "Gradient Boosting":{
            #         'learning_rate':[.1,.01,.05,.001],
            #         'n_estimators': [8,16,32,64,128,256],
            #         'subsample':[0.6,0.7,0.8,0.9],
            #     },
            #     "Linear Regression":{},
            #     "K-Neighbors Classifer":{
            #         'n_neighbors': [5,7,9,11],
            #         'weights': ['uniform','distance'],
            #         'algorithm': ['ball_tree','kd_tree','brute'],
            #     },
            #     "XGBClassifer":{
            #         'learning_rate':[.1,.01,.05,.001],
            #         'n_estimators': [8,16,32,64,128,256],
            #         'max_depth': [3,5,7,10],
            #     },
            #     "CatBoosting Classifer":{
            #         'depth': [6,8,10],
            #         'learning_rate': [0.01, 0.05, 0.1],
            #         'n_estimators': [30, 50, 100],
            #     },
            #     "Adaboost Classifer":{
            #         'learning_rate': [.1,.01,.05,.001],
            #         'n_estimators': [8,16,32,62,128,256],
            #     },
            # }

            model_report: dict=evaluate_models(X_train=X_train, 
                                              y_train=y_train,
                                              X_test=X_test, 
                                              y_test=y_test,
                                              models = models)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square 
        
        except Exception as e:
            raise CustomException(e, sys) # raise an exception if the data is not split

