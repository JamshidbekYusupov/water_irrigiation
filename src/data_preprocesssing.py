import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

log_path = r'C:\Irrigation_Water_Requirement\logging\imputation.log'
logging.basicConfig(
    filename=log_path,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s'
)

logging.info('Preprocessing Pipeline is started working')

class PreprocessingPipeline:

    def __init__(self,df:pd.DataFrame, target:str):

        self.df = df
        self.target = target
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = LabelEncoder()

    def feature_prep(self):

        X = self.df.drop(columns=self.target)
        y = self.df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        num_cols = self.X_train.select_dtypes(include = [np.number]).columns.to_list()
        cat_cols = self.X_train.select_dtypes(exclude = [np.number]).columns.to_list()

        try:
            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='mean'))
            ])
            logging.info('Numerical pipeline is built')
        except Exception:
            logging.error('Error while building numerical pipeline')
            raise

        try:
            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ])
            logging.info('Categorical pipeline is built')
        except Exception:
            logging.error('Error while building categorical pipeline')
            raise

        try:
            self.pipeline = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, num_cols),
                    ('cat', cat_pipe, cat_cols)
                ],
                verbose_feature_names_out=False # do not adding cat_ or num_ prefix
            )

            self.pipeline.set_output(transform='pandas')
            logging.info('ColumnTransformer is built')
        except Exception:
            logging.error('Error while building ColumnTransformer')
            raise

    def fit_transform(self):
        try:
            Train_set = self.pipeline.fit_transform(self.X_train)
            y_train = self.encoder.fit_transform(self.y_train)
            logging.info('Train set fit_transform completed')
            return Train_set, y_train
        except Exception:
            logging.error('Error while fitting train set')
            raise

    def transform(self):
        try:
            Test_set = self.pipeline.transform(self.X_test)
            y_test = self.encoder.transform(self.y_test)
            logging.info('Transform completed on test set')
            return Test_set, y_test
        except Exception:
            logging.error('Error while transforming test set')
            raise