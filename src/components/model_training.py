import os, sys
from src.logger import logging
from xgboost import XGBRegressor
from dataclasses import dataclass
from catboost import CatBoostRegressor
from src.exception import CustomException
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from src.utils import save_object, evaluate_models
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Model Training - Start")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )
            logging.info("Splitting Train & Test Arrays - Complete")
            
            models = {
                "Random Forest": RandomForestRegressor(), 
                "Decision Tree": DecisionTreeRegressor(), 
                "Gradient Boosting": GradientBoostingRegressor(), 
                "Linear Regression": LinearRegression(), 
                "K-Neighbors Regressor": KNeighborsRegressor(), 
                "XGBoost Regressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False), 
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            logging.info("Training & Evaluating Models")

            models_report, models_fitted = evaluate_models(
                X_train = X_train, 
                y_train = y_train, 
                X_test = X_test, 
                y_test = y_test, 
                models = models
            )

            best_model_score = max(list(models_report.values()))

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found!")

            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]

            best_model = models_fitted.get(best_model_name)

            logging.info(
                "Best Model - {} (R2 Score: {})".format(
                    best_model_name, 
                    str(best_model_score)
                )
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model
            )

            logging.info(
                "Best Model Saved at {}".format(
                    self.model_trainer_config.trained_model_file_path
                )
            )
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
