import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
sys.path.append(r'C:\Irrigation_Water_Requirement')
from src.data_preprocesssing import preprocessing_pipeline
raw_data_path = r'Data\Raw_Data\irrigation_prediction.csv'


df = pd.read_csv(raw_data_path)

X = df.drop(['Irrigation_Need'], axis=1)
y = df['Irrigation_Need']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
num_cols = X_train.select_dtypes(include = [np.number]).columns.to_list()
cat_cols = X_train.select_dtypes(exclude = [np.number]).columns.to_list()
dp = preprocessing_pipeline(num_cols=num_cols, cat_cols=cat_cols)

X_train_array = dp.fit_transform(X_train)
X_test_array = dp.transform(X_test)


ord_encoder = OrdinalEncoder()
y_train = ord_encoder.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test = ord_encoder.transform(y_test.values.reshape(-1, 1)).ravel()

filled_data_path = r'Data\Preprocesssed_Data\missed_data'

output_cols = dp.pipeline.get_feature_names_out()

os.makedirs(filled_data_path, exist_ok=True)
out_path = os.path.join(filled_data_path, 'X_train_filled.csv')
X_train_filled = pd.DataFrame(X_train_array, columns=output_cols)
X_train_filled.to_csv(out_path, index = False)

out_path = os.path.join(filled_data_path, 'X_test_filled.csv')
X_test_filled = pd.DataFrame(X_test_array, columns=output_cols)
X_test_filled.to_csv(out_path, index=False)

out_path = os.path.join(filled_data_path, 'y_train.csv')
pd.DataFrame(y_train, columns=['Irrigation_Need']).to_csv(out_path, index=False)

out_path = os.path.join(filled_data_path, 'y_test.csv')
pd.DataFrame(y_test, columns=['Irrigation_Need']).to_csv(out_path, index=False)



