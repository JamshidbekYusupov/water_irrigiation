import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
sys.path.append(r'C:\Irrigation_Water_Requirement')

from src.tuning_pipeline import Tuning_Pipeline
train_path = r'C:\Irrigation_Water_Requirement\Data\engineered_Data\X_train.csv'
test_path = r'C:\Irrigation_Water_Requirement\Data\engineered_Data\X_test.csv'

df = pd.read_csv(r'Data\Raw_Data\irrigation_prediction.csv')

X_train = pd.read_csv(train_path)
X_test = pd.read_csv(test_path)
y_train = pd.read_csv(r'C:\Irrigation_Water_Requirement\Data\engineered_Data\y_train.csv')
y_test = pd.read_csv(r'C:\Irrigation_Water_Requirement\Data\Preprocesssed_Data\missed_data\y_test.csv')
target = 'Irrigation_Need'

models = {
    "DT": DecisionTreeClassifier(random_state=42),
    "Stack_Cls":StackingClassifier(estimators=[
        ('xgb', XGBClassifier(max_depth = 10, learning_rate = 0.1, objective = 'multi:softmax', scale_pos_weight=1, num_class=3)), 
        ('lr', LogisticRegression(n_jobs=-1, max_iter=1000, random_state=42))],
        final_estimator = RandomForestClassifier(random_state=42))
}

params = {
    "DT": {
        "model__max_depth": [3, 5, 10, 15, 20, None],
        "model__min_samples_split": [2, 3, 5, 10],
        "model__min_samples_leaf": [2, 3, 4, 6],
    },

    "Stack_Cls": {
        # XGBoost base estimator
        "model__xgb__n_estimators": [50, 100, 200],
        "model__xgb__max_depth": [3, 5, 8],
        "model__xgb__learning_rate": [0.01, 0.1],

        # Logistic Regression base estimator
        "model__lr__C": [0.01, 0.1, 1, 10],
        "model__lr__solver": ["lbfgs", "liblinear"]
    }
}


for name, model in models.items():
    tune_pipe = Tuning_Pipeline(algorithm=model, name=name, target=target,
                                  X_train = X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    tune_pipe.pipeline_building().pipeline_fit().tune_grid_search(param_grid=params[name]).model_saving().prediction().getting_metrics()
