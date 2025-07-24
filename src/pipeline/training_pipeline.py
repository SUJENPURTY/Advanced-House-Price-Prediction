import os
import sys
import pandas as pd

from src.pipeline.logger import logging
from src.pipeline.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

if __name__ == '__main__':
    try:
        logging.info("Starting California Housing Price Prediction pipeline...")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()  # type: ignore
        print(f"Train data saved at: {train_data_path}")
        print(f"Test data saved at: {test_data_path}")
        logging.info("Data ingestion completed.")

        # Step 2: Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info("Data transformation completed.")

        # Step 3: Model Training
        trainer = ModelTrainer()
        trainer.initate_model_training(train_arr, test_arr)
        logging.info("Model training completed.")

    except Exception as e:
        logging.error("Pipeline execution failed.")
        raise CustomException(e, sys)  # type: ignore
