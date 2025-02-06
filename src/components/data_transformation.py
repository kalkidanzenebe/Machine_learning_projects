# data_transformation.py
import sys
import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from exception import CustomException
from logger import logging
from utils import save_object


class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
    def get_data_transformer_object(self):
        try:
            numerical_features = ["overall", "finishing", "wage", "potential", "reactions", "age"]
            categorical_features = ["nationality", "club", "preferred_foot", "work_rate", "body_type", "position"]
            
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column = "value"
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]
            
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)
            
            train_array = np.c_[X_train_transformed, np.array(y_train)]
            test_array = np.c_[X_test_transformed, np.array(y_test)]
            
            save_object(self.preprocessor_obj_file_path, preprocessing_obj)
            
            return train_array, test_array, self.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)