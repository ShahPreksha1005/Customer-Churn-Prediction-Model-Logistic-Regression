import os
import sys
from src.utils.logger import logging
from sklearn.preprocessing import FunctionTransformer
import typing
from typing import List, Iterator
from ensure import ensure_annotations
import pandas as pd
import numpy as np
from numpy.typing import NDArray


def handling_categorical(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.warning(f'Handling categorical data: {data.columns.tolist()}')
        def is_float(value):
            try:
                float(value)
                return True
            except:
                return False

        for col in data.columns.tolist():
            data.loc[~data[col].apply(is_float), col] = 0
            data[col] = data[col].fillna(data[col].mean())  # Fill NaN values with the mean
            data[col] = data[col].astype('float64')
        return data

    except Exception as e:
        logging.error('Error in handling categorical function: ', e)


def handling_numerical(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.warning(f'Handling numerical data: {data.columns.tolist()}')
        for col in data.columns.tolist():
            data.loc[~data[col].apply(np.isreal), col] = 0
            data[col] = data[col].fillna(data[col].mean())  # Fill NaN values with the mean
            data[col] = data[col].astype('float64')
        return data
    except Exception as e:
        logging.error('Error in handling numerical function: ', e)
