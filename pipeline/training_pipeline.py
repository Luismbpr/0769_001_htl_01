import os
from src.custom_exception import CustomException
from src.logger import get_logger
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import *

logger = get_logger(__name__)

# """
# As soon as someone runs this file it will execute the scripts
# data ingestion,
# data preprocessing
# model traianning
# """

if __name__=="__main__":
    ## 1. Data Ingestion
    data_ingestion =  DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    ## 2. Data Processing
    data_processing = DataProcessor(train_path=TRAIN_FILE_PATH, test_path=TEST_FILE_PATH, processed_dir=PROCESSED_DIR, config_path=CONFIG_PATH)
    data_processing.process()

    ## 3. Model Training
    trainer = ModelTraining(train_path=PROCESSED_TRAIN_DATA_PATH, test_path=PROCESSED_TEST_DATA_PATH, model_output_path=MODEL_OUTPUT_PATH)
    trainer.run()