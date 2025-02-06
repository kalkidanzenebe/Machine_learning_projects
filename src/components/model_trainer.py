import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from exception import CustomException
from logger import logging
from utils import save_object

class ModelTrainer:
    def __init__(self):
        self.model_file_path = os.path.join("artifacts", "model.pkl")
    
    def train_model(self, train_array, test_array):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            logging.info(f"Model Evaluation - R2 Score: {r2:.2f}, MAE: {mae:.2f}")
            save_object(self.model_file_path, model)
            return r2, mae
        except Exception as e:
            raise CustomException(e, sys)