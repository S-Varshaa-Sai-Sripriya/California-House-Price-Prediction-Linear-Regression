import pandas as pd
from sklearn.datasets import fetch_california_housing
from utils.logger import logger
from utils.exception import CustomException
from utils.utils import read_yaml
import os
import sys

def ingest_data():
    try:
        logger.info("Starting data ingestion...")
        
        #Load configuration
        config = read_yaml("config/config.yaml")
        data_path = config["data_path"]
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        #fetch data
        logger.info("Fetching California housing dataset...")
        housing_data = fetch_california_housing()
        df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
        df['target'] = housing_data.target
        
        #save to csv
        df.to_csv(data_path, index=False)
        logger.info(f"Data saved to {data_path}. Shape: {df.shape}")
        return df  
          
    except Exception as e:
        logger.error("Error during data ingestion", exc_info=True)
        raise CustomException(str(e), sys)