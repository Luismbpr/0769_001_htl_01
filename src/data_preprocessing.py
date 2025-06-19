import os
import numpy as np
import pandas as  pd

## When Running all scripts
#from src.logger import get_logger
#from src.custom_exception import CustomException

## When running this standalone script
from .logger import get_logger
from .custom_exception import CustomException

from config.paths_config import *
from utils.common_functions import read_yaml, load_data

## RandomForestClassifier for feature selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

# RAW_FILE_PATH
# TRAIN_FILE_PATH
# TEST_FILE_PATH
# PROCESSED_DIR
# PROCESSED_TRAIN_DATA_PATH
# PROCESSED_TEST_DATA_PATH

## config.yaml - Need to read this
# data_processing
# categorical_columns
# numerical_columns
class DataProcessor:
    def __init__(self, train_path:str, test_path:str, processed_dir:str, config_path:str):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(file_path=config_path)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)
    
    def preprocess_data(self, df):
        try:
            #pass
            logger.info("Data Processing Process Started")
            
            ## Dropping Columns
            logger.info("Dropping the columns")
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True)
            logger.info(f"df Shape after dropping columns: {df.shape}")

            ## Dropping missing/null values if there are (rows)
            logger.info("Dropping null values")
            df.dropna(axis=0, inplace=True)
            logger.info(f"df Shape after dropping null values: {df.shape}")

            ## Dropping duplicated values if there are
            logger.info("Dropping duplicated values")
            df.drop_duplicates(inplace=True)
            logger.info(f"df Shape after dropping duplicated values: {df.shape}")

            ## Feature Separation Categorical and Numerical cols
            ## Info taken from config/config.yaml
            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            ## LabelEncoder
            logger.info("Applying Label Encoding")
            label_encoder = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label:code for label,code in zip(label_encoder.classes_ , label_encoder.transform(label_encoder.classes_))}
            
            logger.info("Label Mappings are: ")
            for col, mapping in mappings.items():
                logger.info(f"{col}: {mapping}")
            
            ## Check for multicollinearity (Skipped since this data did not have)
            
            ## Handling Skewness
            logger.info("Handling Skewness in Features")
            
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            ## Lambda function to store .skew in skewness variable for all num columns
            skewness = df[num_cols].apply(lambda x:x.skew())
            
            ## Apply a log transformation to fix the skewness if the skewness is greater than the threshold
            for column in skewness[skewness>skew_threshold].index:
                df[column] = np.log1p(df[column])
            
            return df

        except Exception as e:
            logger.error(f"Error during preprocessing step {e}")
            raise CustomException("Error on the preprocessing data process", e)
    

    ## Handling Imbalanced Data
    def balance_data(self, df):
        """
        Method for handling Imbalanced data
        - Separate into X, y
        - Performing Oversampling Technique
        - New DataFrame Creation with balanced data

        """
        try:
            #pass
            logger.info("Handling Imbalanced Data")
            X = df.drop(columns='booking_status')
            y = df["booking_status"]

            ## Oversampling Technique
            smote = SMOTE(random_state=42)
            X_resampled , y_resampled = smote.fit_resample(X,y)

            balanced_df = pd.DataFrame(X_resampled , columns=X.columns)
            balanced_df["booking_status"] = y_resampled

            logger.info("Successfully Balanced Data")
            return balanced_df
        
        except Exception as e:
            logger.error(f"Error during Data Balancing step {e}")
            raise CustomException("Error while data balancing", e)
    
    ## Feature Selection
    def select_features(self, df):
        """
        Feature Selection Process
        Separating into X, y to train the model
        Using RandomForestClassifier to select the best features
        Create a new DF with the selected features and the target variable
        """
        try:
            #pass
            logger.info("Feature Selection Process Started")
            #X = df.drop(columns=["booking_status"])
            X = df.drop(columns=["booking_status"], axis=1)
            y = df["booking_status"]

            ## Training on all data.
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature':X.columns,
                                                  'importance':feature_importance
                                                  })
            top_features_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

            num_features_to_select = self.config["data_processing"]["no_of_features"]

            top_x_features = top_features_importance_df["feature"].head(num_features_to_select).values
            
            logger.info(f"Features Selected: {top_x_features }")

            top_x_df = df[top_x_features.tolist() + ["booking_status"]]

            logger.info("Feature Selection Completed Successfully")

            return top_x_df
        
        except Exception as e:
            logger.error(f"Error during Feature Selection step {e}")
            raise CustomException("Error on the Feature Selection process", e)
    
    ## Data Processing Is Complete now I need to save it on csv format
    def save_data(self, df, file_path):
        """
        Saving Processed data into csv Format
        """
        try:
            #pass
            logger.info("Saving Data Process Started")
            df.to_csv(file_path, index=False)

            logger.info(f"Data Saved Successfully to path:\n{file_path}")
        
        except Exception as e:
            logger.error(f"Error during Data Saving step {e}")
            raise CustomException("Error on Data Saving process", e)
    

    ## Process
    def process(self):
        try:
            #pass
            logger.info("Loading data from RAW Directory")
            
            ## load_data function from utils/common_functions.py
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)
            
            train_df = self.balance_data(df=train_df)
            test_df = self.balance_data(df=test_df)
            
            ## Only for train_df since we want to have the same features for both train and test_df
            train_df = self.select_features(df=train_df)
            test_df = test_df[train_df.columns]
            
            self.save_data(df=train_df, file_path=PROCESSED_TRAIN_DATA_PATH)
            self.save_data(df=test_df, file_path=PROCESSED_TEST_DATA_PATH)
            
            logger.info("Data Processing Completed Successfully")
        
        except Exception as e:
            logger.error(f"Error during Feature Selection step {e}")
            raise CustomException("Error on the Feature Selection process", e)


if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()