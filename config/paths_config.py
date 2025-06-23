import os

######################### Data Ingestion #########################
RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "config/config.yaml"


######################### Data Processing 
# config/paths_config.py #########################
## Removing Duplicates
## Replacing NaN values
## Encoding - LabelEncoder
## Reducing data skewness df.skew()
## Over Sampling SMOTE - from imblearn.over_sampling import SMOTE
PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")

######################### Experiment Tracking and Model Training #########################
# config/paths_config.py
MODEL_DIR = "artifacts/models"
MODEL_OUTPUT_NAME = os.path.join(MODEL_DIR, "lgbm_model.pkl")
MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"


######################### . #########################