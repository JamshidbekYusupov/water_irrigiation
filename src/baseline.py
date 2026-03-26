import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import logging
import os
import joblib
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

log_path = r'C:\Irrigation_Water_Requirement\logging\basline.log'
logging.basicConfig(
    filename=log_path,
    filemode='a',
    level=logging.INFO,
    format = '%(asctime)s-%(levelname)s-%(message)s'
)

logging.info(f'Basline model training has started')

class BaselinePipeline(BaseEstimator):

    def __init__(self, algorithm, name, target:str, X_train, X_test, y_train, y_test):

        # self.df = df
        self.X_test = X_test
        self.X_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = None
        self.metrics = {}
        self.algorithm = algorithm
        self.model_name = name
        self.preprocessor = None
        self.model = None

    def pipeline_building(self):
        try:

            num_cols = self.X_train.select_dtypes(include = [np.number]).columns.to_list()
            cat_cols = self.X_train.select_dtypes(exclude = [np.number]).columns.to_list()

            self.X_train[cat_cols] = self.X_train[cat_cols].astype(str)
            self.X_test[cat_cols] = self.X_test[cat_cols].astype(str)

            cat_pipe = Pipeline([
                ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            num_pipe = Pipeline([
                ('scalar', StandardScaler())
                ])
            
            self.preprocessor = ColumnTransformer([
                ('num', num_pipe, num_cols),
                ('cat', cat_pipe, cat_cols)
            ])
            logging.info(f'Pipeline for categorical and numerical features is built')
            return self
        except Exception as e:
            logging.error(f'Error while building pipeline for categorical and numerical features')
            raise
    
    def pipeline_fit(self):
        try:

            if self.algorithm is None:
                raise ValueError('Model algorithm is not specified, please specify model first')
            
            self.model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', self.algorithm)
            ])

            self.model.fit(self.X_train, self.y_train)
            logging.info(f'Model is trained with {self.model_name}')

            return self
        except Exception as e:
            logging.error(f'Error while training model with {self.model_name} algorithm, Error: {e}')
            raise
   
# Saving model to the models folder
    def model_saving(self):
        try:
            out_dir = r'C:\Irrigation_Water_Requirement\Models\all_models'
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'{self.model_name}.joblib')
            joblib.dump(self.model, out_path)
            logging.info(f'{self.model_name} is saved at {out_dir}')
            return self
        except Exception as e:
            logging.error(f'Error while saving {self.model_name}')
            raise
    def prediction(self):
        try:
            self.y_pred = self.model.predict(self.X_test)
            logging.info(f'Prediction is done with {self.model_name}')
            return self
        except Exception as e:
            logging.info(f'Error while predcting with {self.model_name}')
            raise
    def getting_metrics(self):
        try:
            # self.metrics[f'{self.model_name}_accuracy'] = accuracy_score(self.y_test, self.y_pred)
            # self.metrics[f'{self.model_name}_precison'] = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
            # self.metrics[f'{self.model_name}_f1_score'] = f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
            # self.metrics[f'{self.model_name}_recall'] = recall_score(self.y_test, self.y_pred,average='weighted', zero_division=0)

            self.metrics = {
                'Model:': self.model_name,
                'Accuracy Score:': accuracy_score(self.y_test, self.y_pred),
                'Precision Score:': precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0),
                'F1_score': f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(self.y_test, self.y_pred,average='weighted', zero_division=0)
            }

            metrics_df = pd.DataFrame([self.metrics])
            metrics_path = r'C:\Irrigation_Water_Requirement\Metrics'
            os.makedirs(metrics_path, exist_ok=True)
            out_path = os.path.join(metrics_path, 'Basline_results.txt')
            table = tabulate(metrics_df, headers='keys', tablefmt='grid', showindex=True)
            with open(out_path, 'a') as f:
                f.write(f"Evaluation Results of {self.model_name}\n")
                f.write(table)
                f.write('\n')
            logging.info(f'Results are saved at {self.model_name}')
            return self
        except Exception as e:
            logging.error(f'Error while saving results: {e}')
            raise
