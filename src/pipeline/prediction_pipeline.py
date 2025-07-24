import sys
import os
import pandas as pd
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import load_object

# Prediction Pipeline
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)  # type: ignore


# CustomData class for California Housing dataset
class CustomData:
    def __init__(self, longitude, latitude, housing_median_age, total_rooms,
                 total_bedrooms, population, households, median_income,
                 ocean_proximity):
        self.longitude = longitude
        self.latitude = latitude
        self.housing_median_age = housing_median_age
        self.total_rooms = total_rooms
        self.total_bedrooms = total_bedrooms
        self.population = population
        self.households = households
        self.median_income = median_income
        self.ocean_proximity = ocean_proximity

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "longitude": [self.longitude],
                "latitude": [self.latitude],
                "housing_median_age": [self.housing_median_age],
                "total_rooms": [self.total_rooms],
                "total_bedrooms": [self.total_bedrooms],
                "population": [self.population],
                "households": [self.households],
                "median_income": [self.median_income],
                "ocean_proximity": [self.ocean_proximity]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)  # type: ignore



