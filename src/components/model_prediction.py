import os 
import sys
import pandas as pd
from src.utils.logger import logging
from src.utils.utils import load_pipeline, save_artifacts
from src.config.configuration import ModelPredictionConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelPrediction:
    def __init__(self, config: ModelPredictionConfig=None):
        self.config = config
        self.model = None
        self.test_data = None
        self.test_features = None
        self.test_target = None
        self.predictions = None

    def run_pipeline(self):
        try:
            logging.info("Model Prediction Started")
            self.test_data = pd.read_csv(self.config.test_data_path)
            self.model = load_pipeline(pipeline_name='model')
            self.test_features = self.test_data.drop(columns=['Outcome'])
            self.test_target = self.test_data['Outcome']

            predictions_array = self.model.predict(self.test_features)
            self.predictions = pd.Series(predictions_array, name='predictions')
            save_artifacts(object=self.predictions, file_name='predictions', file_type='csv')
            
            logging.info("Model Prediction Completed")

        except Exception as e:
            logging.error(f'Error at model prediction run pipeline: {e}')

    def get_results(self):
        try:
            logging.info("Getting results")

            accuracy = accuracy_score(self.test_target, self.predictions)
            precision = precision_score(self.test_target, self.predictions)
            recall = recall_score(self.test_target, self.predictions)
            f1 = f1_score(self.test_target, self.predictions)

            logging.critical(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
            
        except Exception as e:
            logging.error(f'Error at getting results: {e}')

    @staticmethod
    def make_predictions(data=None):
        try:
            logging.info("Making predictions started")
            model = load_pipeline(pipeline_name='model')
            new_predictions = model.predict(data)

            logging.critical(f'New predictions: {new_predictions}')
            return new_predictions
        except Exception as e:
            logging.error(f'Error at making predictions: {e}')
