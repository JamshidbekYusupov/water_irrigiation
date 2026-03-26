import os, sys
import pandas as pd
import numpy as np

sys.path.append(r'C:\Irrigation_Water_Requirement')
from src.feature_engineering import Engineerig

train_path = r'C:\Irrigation_Water_Requirement\Data\Preprocesssed_Data\missed_data\X_train_filled.csv'
test_path = r'C:\Irrigation_Water_Requirement\Data\Preprocesssed_Data\missed_data\X_test_filled.csv'
# DF 
X_train = pd.read_csv(train_path)
X_test = pd.read_csv(test_path)
y_train = pd.read_csv(r'C:\Irrigation_Water_Requirement\Data\Preprocesssed_Data\missed_data\y_train.csv')

eng = Engineerig(X_train=X_train,X_test=X_test, y_train=y_train)
eng.feature_adding().handle_skewness().imbalance_handling().data_saving()