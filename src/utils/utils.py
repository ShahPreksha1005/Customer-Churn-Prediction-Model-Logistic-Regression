import os
import sys
from src.utils.logger import logging
from ensure import ensure_annotations
import typing
from typing import Optional, Union
from pathlib import Path
import pandas as pd
import types
import sklearn.pipeline
import joblib

@ensure_annotations
def make_dir(folder_name: str, parent_location: bool , current_location: Path = Path(os.getcwd())):
    try:
        if parent_location:
            folder = os.path.join(os.getcwd(), folder_name)
        else:
            folder = os.path.join(current_location, folder_name)
        os.makedirs(folder, exist_ok=True)
        logging.critical(f'{folder_name} directory created successfully.')
    except Exception as e: 
        logging.warning('Exception occurred while creating directory: ', e)

def save_artifacts(object: Union[pd.DataFrame, sklearn.pipeline.Pipeline], file_name: str, file_type: str):
    try:
        if file_type == 'csv':
            file = os.path.join('Artifacts', file_name+'.csv')
            object.to_csv(file, index=False)
            logging.critical(f'{file_name} is stored in the artifacts folder.')
        elif file_type == 'joblib':
            joblib.dump(object, os.path.join('Artifacts', file_name+'.joblib'))
            logging.critical(f'{file_name} is stored in the artifacts folder.')
    except Exception as e:
        logging.error('Error occurred while storing the data in the artifacts folder: ', e)

@ensure_annotations
def load_pipeline(pipeline_name: str):
    try:
        return joblib.load(os.path.join('Artifacts', pipeline_name+'.joblib')) 
    except Exception as e:
        logging.error('Error occurred while loading the pipeline from the artifacts folder: ', e)
