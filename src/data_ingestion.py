import os
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage
#from src.logger import get_logger
#from src.custom_exception import CustomException
from .logger import get_logger
from .custom_exception import CustomException
#from logger import get_logger
#from custom_exception import CustomException


## Import all paths from paths_config.py, able to do this due to __init__.py
from config.paths_config import *
#from config.paths_config import RAW_DIR,RAW_FILE_PATH

from utils.common_functions import read_yaml
#from utils.common_functions import load_data


logger = get_logger(__name__)

###        """
###        Getting the information paths from config/config.yaml
###
###        data_ingestion:
###          bucket_name : "my_bucket9789"
###          bucket_file_name : "Hotel_Reservations.csv"
###          train_ratio : 0.8
###        
###        Create the directories required from config/paths_config.py
###
###        Creating Storing the ra
###        """

class DataIngestion:
    def __init__(self, config):
        self.config = config['data_ingestion']
        self.bucket_name = self.config['bucket_name']
        self.file_name = self.config['bucket_file_name']
        self.train_ratio = self.config['train_ratio']

        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Data Ingestion started with {self.bucket_name} and file is {self.file_name}")
    
        # """
        # Gets the data from the GCP Bucket and saves it locally
        # """
    def download_csv_from_gcp(self):
        try:
            #pass
            #client = storage.Client()
            #bucket = client.bucket(self.bucket_name)
            ## blob is the filename
            #blob = bucket.blob(self.file_name)
            #blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"CSV file successfully downloaded to {RAW_FILE_PATH}")
        
        except Exception as e:
            logger.error("Error while downnloading the csv file")
            raise CustomException("Failed to download the csv file", e)
    
        # """
        # Split data from RAW_FILE_PATH
        # """
    def split_data(self):
        try:
            logger.info("Starting Data Split Process")
            data = pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(data, test_size=1-self.train_ratio, random_state=42)
            ## This will be in DataFrame format and want them in CSV format
            train_data.to_csv(TRAIN_FILE_PATH)#, header=True, index=False)
            test_data.to_csv(TEST_FILE_PATH)#, header=True, index=False)
            
            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")
        
        except Exception as e:
            logger.error("Error while splitting the data")
            raise CustomException("Failed to split the data", e)
    
        # """
        # Runs the DataIngestion and its methods
        # """
    def run(self):
        try:
            #pass
            logger.info("Starting Data Ingestion Process")

            self.download_csv_from_gcp()
            self.split_data()

            logger.info("Data Ingestion Completed Successfully")
        
        except Exception as e:
            logger.error(f"CustomException: {str(e)}")
            #raise CustomException(e)
        finally:
            logger.info("Data Ingestion Completed")


if __name__=="__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    #data_ingestion = DataIngestion(read_yaml(RAW_FILE_PATH))
    data_ingestion.run()
### Weightless Context awareness

## Note: When running this standalone script
### Use this:
##from .logger import get_logger
##from .custom_exception import CustomException
##% python -m src.data_ingestion
## This worked