import os
import sys
from src.utils.logger import logging
from src.config.configuration import Configuration
from src.components.data_preparation import DataPreparation

class DataPreparationPipeline:

    def __init__(self):
        logging.info('Started data preparation pipeline..!!')


    def main(self):

        try:
            logging.info("Data Preparation Pipeline Started")
            config = Configuration()
            preparation_config = config.data_preparation_config()
            data_preparation = DataPreparation(config=preparation_config)
            data_preparation.create_pipeline()
            data_preparation.run_pipeline()
        except Exception as e:
            logging.error(f'Error in Data Preparation Pipeline: {e}')
