import os
import numpy as np
import pandas as  pd
import joblib
import lightgbm as lgbm

## When Running all scripts
#from src.logger import get_logger
#from src.custom_exception import CustomException

## When running this standalone script
from .logger import get_logger
from .custom_exception import CustomException

from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint

import mlflow
import mlflow.sklearn


logger = get_logger(__name__)

class ModelTraining():
    def __init__(self, train_path:str, test_path:str, model_output_path:str):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
        
        
        #self.model_output_name = MODEL_OUTPUT_NAME##Verify if this works
        
        ## Verify if this creates the folder
        # if not os.path.exists(self.model_output_path):
        #     os.makedirs(self.model_output_path, exist_ok=True)
        
        self.model_dir = MODEL_DIR
        ## Verify if this creates the folder
        if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir, exist_ok=True)
    

    def load_and_split(self):
        """
        Args


        Returns
          X_train, y_train, X_test, y_test
        """
        try:
            #pass
            logger.info(f"Loading data from:\n{self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from:\n{self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            #logger.info(f"Sequence Verification for X_train:\n{X_train.columns}")
            #logger.info(f"Sequence Verification for X_test:\n{X_test.columns}")
            
            X_train_cols = list[X_test.columns]
            X_test_cols = list[X_test.columns]

            if X_train_cols == X_test_cols:
                #logger.info(f"Sequence Verification Successful")
                logger.info(f"Columns are on the same sequence in both Train and Test Sets.")
            else:
                #logger.info(f"Sequence Verification Unsuccessful")
                logger.info(f"Columns are not on the same sequence in Train and Test Sets.")
            
            logger.info("Data Split for Model Training Successful")

            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load data", e)
    

    def train_lgbm(self, X_train, y_train):
        """
        """
        try:
            #pass
            logger.info("Initializing Model Trainer")
            
            #lgbm_model = lgbm.LGBMClassifier(random_state=self.params_dist["random_state"])
            
            lgbm_model = lgbm.LGBMClassifier(random_state=self.random_search_params["random_state"])

            logger.info("Starting Hyperparameter Tuning")

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter = self.random_search_params["n_iter"],
                cv = self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )
            
            logger.info("Training The Model")

            random_search.fit(X_train,y_train)

            logger.info("Hyperparameter Tuning Completed")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best Model: {best_lgbm_model}")
            logger.info(f"Best Parameters: {best_params}")
            
            logger.info("Model Trainer Completed Successfully")

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to train model",  e)
        
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Model Evaluation on Unseen data.
        
        Performs model prediction on unseen data and then calculates required metrics.

        Args
          model
          X_test
          y_test
        
        Returns
          Dict with Accuracy, Precision, Recall and F-1 scores.
        """
        try:
            #pass
            logger.info("Model Evaluation Started")
            
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Metric - Accuracy Score: {accuracy}")
            logger.info(f"Metric - Precision Score: {precision}")
            logger.info(f"Metric - Recall Score: {recall}")
            logger.info(f"Metric - F1 Score: {f1}")
            
            logger.info("Model Evaluation Completed Successfully")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        except Exception as e:
            logger.error(f"Error while evaluating model {e}")
            raise CustomException("Failed to evaluate model",  e)
    

    def save_model(self, model):
        """
        """
        try:
            #pass
            logger.info(f"Creating output path directory if it does not exist")
            
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            logger.info(f"Saving model to path:\n{self.model_output_path}")
            joblib.dump(model, self.model_output_path)
            logger.info("Model Saved Complete")
        
        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to save model",  e)
    

    def run(self):
        """
        """
        try:
            with mlflow.start_run():
                logger.info("Model Training Pipeline Started")
                
                logger.info("Starting MLFlow Experimentation")

                ## Logging dataset (single artifact)
                logger.info("Logging the Training And Testing Dataset to MLFlow")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")
                
                X_train,y_train,X_test,y_test =self.load_and_split()
                best_lgbm_model = self.train_lgbm(X_train,y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(model=best_lgbm_model)

                logger.info("Logging the model into MLFlow")
                mlflow.log_artifact(self.model_output_path)

                ## log_artifacts because there are several parameters and metrics
                logger.info("Logging Parameters and Metrics to MLFlow")
                mlflow.log_params(best_lgbm_model.get_params())
                #mlflow.log_artifacts(metrics)
                #mlflow.log_dict(metrics, "metrics")
                mlflow.log_metrics(metrics)
                
                logger.info("Model Training Pipeline Completed")
        except Exception as e:
            logger.error(f"Error in Model Training Pipeline {e}")
            raise CustomException("Failed During Model Training Pipeline",  e)


if __name__ == "__main__":
    trainer = ModelTraining(train_path=PROCESSED_TRAIN_DATA_PATH, test_path=PROCESSED_TEST_DATA_PATH, model_output_path=MODEL_OUTPUT_PATH)
    trainer.run()