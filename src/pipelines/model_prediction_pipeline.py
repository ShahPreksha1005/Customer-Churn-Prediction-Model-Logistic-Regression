import os
import sys
import pandas as pd
import numpy as np
from src.utils.logger import logging
from src.config.configuration import Configuration
from src.components.model_prediction import ModelPrediction

class ModelPredictionPipeline:
    def __init__(self):
        print('started model predictions pipeline..!!')

    def main(self):
        try:
            logging.info("Model Prediction Pipeline Started")
            config = Configuration()
            model_prediction_config = config.model_prediction_config()
            model_prediction = ModelPrediction(config=model_prediction_config)
            model_prediction.run_pipeline()
            
            # Modify the values for diabetes prediction
            values = [2, 138, 62, 35, 0, 33.6, 0.127, 47]
            values_array = np.array(values).reshape(1, 8)
            temp_series = pd.DataFrame(values_array, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
            
            # Make predictions
            result = model_prediction.make_predictions(temp_series)
            logging.critical(f'Prediction result: {result}')

            # Get prediction results
            model_prediction.get_results()

            logging.info('Model Prediction Pipeline Completed')
        except Exception as e:
            logging.error(f'Error at model prediction run pipeline: {e}')
