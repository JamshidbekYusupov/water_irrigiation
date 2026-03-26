import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
sys.path.append(r'C:\Irrigation_Water_Requirement')

from src.baseline import BaselinePipeline
train_path = r'C:\Irrigation_Water_Requirement\Data\Preprocesssed_Data\missed_data\X_train_filled.csv'
test_path = r'C:\Irrigation_Water_Requirement\Data\Preprocesssed_Data\missed_data\X_test_filled.csv'

df = pd.read_csv(r'Data\Raw_Data\irrigation_prediction.csv')

X_train = pd.read_csv(train_path)
X_test = pd.read_csv(test_path)
y_train = pd.read_csv(r'C:\Irrigation_Water_Requirement\Data\Preprocesssed_Data\missed_data\y_train.csv')
y_test = pd.read_csv(r'C:\Irrigation_Water_Requirement\Data\Preprocesssed_Data\missed_data\y_test.csv')
target = 'Irrigation_Need'

models = {
    "DT": DecisionTreeClassifier(random_state=42),
    "Stack_Cls":StackingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=42, n_estimators=15)), 
        ('xgb', XGBClassifier(max_depth = 10, learning_rate = 0.1, objective = 'multi:softmax', scale_pos_weight=1, num_class=3)), 
        ('lr', LogisticRegression(n_jobs=-1, max_iter=1000, random_state=42))],
        final_estimator = DecisionTreeClassifier(random_state=42))
}


for name, model in models.items():
    basic_pipe = BaselinePipeline(algorithm=model, name=name, target=target,
                                  X_train = X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    basic_pipe.pipeline_building().pipeline_fit().model_saving().prediction().getting_metrics()
