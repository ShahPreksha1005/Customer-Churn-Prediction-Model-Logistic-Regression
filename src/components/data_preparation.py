import pandas as pd
from src.utils.logger import logging
from src.config.configuration import Configuration
from pathlib import Path
from ensure import ensure_annotations
import typing
from src.components.transformers import handling_categorical, handling_numerical
from typing import List
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.utils import load_pipeline, save_artifacts


class DataPreparation:

    def __init__(self, config: Configuration):
        self.config = config
        self.data = None
        self.features = None
        self.target = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.required_columns = None
        
    def create_pipeline(self):
        try:
            logging.critical("Data Preparation Started")
            self.data = pd.read_csv(self.config.raw_data_path)
            
            # Considering only required columns
            self.required_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 
                                     'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                     'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                                     'MonthlyCharges', 'TotalCharges', 'Churn']
            self.data = self.data[self.required_columns]

            self.features = self.data.drop(columns=['Churn'])
            self.target = self.data['Churn']

            self.categorical_columns = self.features.select_dtypes(include=['object']).columns.tolist()
            self.numerical_columns = self.features.select_dtypes(exclude=['object']).columns.tolist()
            
            logging.warning(f'Categorical columns: {self.categorical_columns}')
            logging.warning(f'Numerical columns: {self.numerical_columns}')
            
            # Creating pipeline
            categorical_transformer = FunctionTransformer(handling_categorical)
            numerical_transformer = FunctionTransformer(handling_numerical)

            preprocessor = ColumnTransformer(
                transformers=[
                    ('categorical', categorical_transformer, self.categorical_columns),
                    ('numerical', numerical_transformer, self.numerical_columns)
                ])

            preparation_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor)
            ])

            save_artifacts(object=preparation_pipeline, file_name=self.config.pipeline_name, file_type='joblib')
            logging.info("Data Preparation Completed")

        except Exception as e:
            logging.error(f'Error at create pipeline function: {e}')


    def run_pipeline(self):
        try:
            preparation_pipeline = load_pipeline(self.config.pipeline_name)
            logging.info("Data Preparation Pipeline Started")
            data_array = preparation_pipeline.fit_transform(self.features)
            new_columns = self.categorical_columns + self.numerical_columns
            
            new_features = pd.DataFrame(data=data_array, columns=new_columns)
            
            cleaned_data = pd.concat([new_features, self.target], axis=1)
            save_artifacts(object=cleaned_data, file_name='cleaned_data', file_type='csv')
            logging.info("Data Preparation Pipeline Completed")
        except Exception as e:
            logging.error(f'Error at run pipeline function: {e}')
