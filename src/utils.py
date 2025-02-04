import os 
import sys

import numpy as np
import pandas as pd
import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train,y_train, X_test,y_test,models):
    try: 
        report = {}

        # for model_name, model in models.items():
        #     # get parameters for current model
        #     para = params[model_name]

        #     # if parameters exist for the model , perform GridSearchCV
        #     if para:
        #         gs = GridSearchCV(model, para, cv=3)
        #         gs.fit(X_train, y_train)

        #     # use best parameters for the model
        #     model.set_params(**gs.best_params_)

        #     # Make predictions
        #     y_train_pred = model.predict(X_train) # Train Model
        #     y_test_pred = model.predict(X_test)


        #     # get R2 scores
        #     train_model_score = r2_score(y_train, y_train_pred)
        #     test_model_score = r2_score(y_test, y_test_pred)

        #     report[model_name] = test_model_score

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train) # Train Model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)