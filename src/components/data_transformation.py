import sys, os
import numpy as np
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    

    def get_data_transformer_object(self):
        try:
            num_cols = [
                "writing_score", 
                "reading_score"
            ]

            cat_cols = [
                "gender", 
                "race_ethnicity", 
                "parental_level_of_education", 
                "lunch", 
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one-hot_encoding", OneHotEncoder()), 
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, num_cols), 
                    ("categorical_pipeline", cat_pipeline, cat_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation - Start")
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logging.info("Train & Test Dataset's Loading - Complete")

            logging.info("Obtaining Data Transformer's Pipeline Object")
            preprocessing_obj = self.get_data_transformer_object()

            target_col = "math_score"

            logging.info("Creating Input & Target Sub-Datasets for both Train & Test Datasets")
            input_feature_train_df = df_train.drop(columns=[target_col], axis=1)
            target_feature_train_df = df_train[target_col]

            input_feature_test_df = df_test.drop(columns=[target_col], axis=1)
            target_feature_test_df = df_test[target_col]

            logging.info("Applying Data Transformation")
            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            """`numpy.c_`:
            It's an indexing utility in NumPy that facilitates the concatenation of arrays along the second axis (columns). 
            It is a convenient shorthand for stacking arrays as columns, particularly useful when dealing with 1-D arrays 
            that need to be treated as column vectors in a 2-D array."""

            train_array = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            logging.info("Saving Preprocessing Object")
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path, 
                preprocessing_obj
            )
            logging.info("Data Transformation - Complete")

            return (
                train_array, 
                test_array, 
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
