import logging
import os
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

#LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.log")
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%M-%d')}.log")

## Logger Configuration
## 3 params required filename, format, level
## 3 main levels are info for information messages, warning for warning messages and error for error messages
logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

    # """
    # Function to initialize the logger in different files
    # It will create a logger with the desired name

    # Args
    #   name:str - Name of the logger
    
    # """
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger