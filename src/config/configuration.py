import os
import sys
from src.utils.logger import logging
from dataclasses import dataclass
from src.entity_config.entity_config import DataIngestionConfig, DataPreparationConfig, ModelTrainingConfig, ModelPredictionConfig
import yaml
from ensure import ensure_annotations
from pathlib import Path
from typing import Union

@dataclass
class Configuration:
    config_file_path: Union[str, Path] = 'src/config/config.yaml'

    def __post_init__(self):
        self.configuration = self.load_configuration()

    def load_configuration(self):
        try:
            with open(self.config_file_path, 'r') as stream:
                return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(f'Error in loading yaml file: {exc}')

    @ensure_annotations
    def data_ingestion_config(self) -> DataIngestionConfig:
        try:
            ingestion_params = self.configuration['data_ingestion']
            return DataIngestionConfig(raw_data_path=ingestion_params['raw_data_path'])
        except KeyError as e:
            logging.error(f'Error at data ingestion config: {e}')

    @ensure_annotations
    def data_preparation_config(self) -> DataPreparationConfig:
        try:
            preparation_params = self.configuration['data_preparation']
            return DataPreparationConfig(
                raw_data_path=preparation_params['raw_data'],
                pipeline_name=preparation_params['pipeline_name']
            )
        except KeyError as e:
            logging.error(f'Error at data preparation config: {e}')

    @ensure_annotations
    def model_training_config(self) -> ModelTrainingConfig:
        try:
            model_training_params = self.configuration['model_training']
            return ModelTrainingConfig(
                cleaned_data_path=model_training_params['cleaned_data_path'],
                test_size=model_training_params['test_size']
            )
        except KeyError as e:
            logging.error(f'Error at model training config: {e}')

    @ensure_annotations
    def model_prediction_config(self) -> ModelPredictionConfig:
        try:
            model_prediction_params = self.configuration['model_prediction']
            return ModelPredictionConfig(
                test_data_path=model_prediction_params['test_data_path']
            )
        except KeyError as e:
            logging.error(f'Error at model prediction config: {e}')
