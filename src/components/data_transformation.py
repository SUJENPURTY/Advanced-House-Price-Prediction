from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys, os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import save_object

## Configuration for saving preprocessor
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Columns in California housing dataset
            numerical_cols = [
                'longitude', 'latitude', 'housing_median_age',
                'total_rooms', 'total_bedrooms', 'population',
                'households', 'median_income'
            ]

            logging.info('Pipeline Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols)
                ]
            )

            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            logging.error("Error in Data Transformation")
            raise CustomException(e, sys)  # type: ignore

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head:\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:\n{test_df.head().to_string()}')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'median_house_value'

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved.')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)  # type: ignore
