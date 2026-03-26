import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import logging
log_path = r'C:\Irrigation_Water_Requirement\logging\imputation.log'
logging.basicConfig(
    filename=log_path,
    filemode='a',
    level=logging.INFO,
    format = '%(asctime)s-%(levelname)s-%(message)s'
)

logging.info('Preprocessing Pipeline is started working')

class preprocessing_pipeline:

    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        try:

            # Building a imputing pipeline
            self.num_pipe = Pipeline([
                ('num', SimpleImputer(strategy='mean'))
            ])

            logging.info(f'Numerical Pipeline is built')
        except Exception as e:
            logging.error(f'Error while building numerical pipeline')
            raise
        try:

            self.cat_pipe = Pipeline([
                ('cat', SimpleImputer(strategy='most_frequent'))
            ])
            logging.info(f'Categorical pipeline is built')
        except Exception as e:
            logging.error(f'Error while building categorical pipeline')
            raise

        try:
            self.pipeline = ColumnTransformer([
                ('num', self.num_pipe, self.num_cols),
                ('cat', self.cat_pipe, self.cat_cols)
            ])
            logging.info(f'Column Transformer is built')
        except Exception as e:
            logging.error(f'Error while building column transformer')
            raise

    def fit_transform(self, X, y = None):
        try:
            transformed = self.pipeline.fit_transform(X)
            logging.info(f'Train Set is fitted')
            return transformed
        except Exception as e:
            logging.error(f'Error while train sat fitting')
            raise
    def transform(self, X):
        try:
            transformed = self.pipeline.transform(X)
            logging.info(f'Transform is done to test set')
            return transformed
        except Exception as e:
            logging.error(f'Error while doing transform on test set')
            raise
    