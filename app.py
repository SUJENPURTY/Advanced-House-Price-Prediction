import os
import sys
from flask import Flask, request, render_template 
import pandas as pd

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.pipeline.logger import logging
from src.pipeline.exception import CustomException

app = Flask(__name__)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            # Read input from the form for California Housing
            data = CustomData(
                longitude=float(request.form.get('longitude')),  # type: ignore
                latitude=float(request.form.get('latitude')),    # type: ignore
                housing_median_age=float(request.form.get('housing_median_age')),  # type: ignore
                total_rooms=float(request.form.get('total_rooms')),  # type: ignore
                total_bedrooms=float(request.form.get('total_bedrooms')),  # type: ignore
                population=float(request.form.get('population')),  # type: ignore
                households=float(request.form.get('households')),  # type: ignore
                median_income=float(request.form.get('median_income')),  # type: ignore
                ocean_proximity=request.form.get('ocean_proximity')  # Categorical (string) # type: ignore
            )

            # Convert input data to DataFrame
            final_new_data = data.get_data_as_dataframe()
            logging.info(f"Input DataFrame:\n{final_new_data}")

            # Make prediction
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(final_new_data)

            logging.info(f"Prediction Output: {prediction}")

            # Round predicted housing price
            result = round(float(prediction[0]), 2)

            return render_template('form.html', result=result)

        except Exception as e:
            raise CustomException(e, sys)  # type: ignore

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
