# As I am using tree based algorithms,we need to use encoded set for enginnering
# 
#  Added features

# ~Total_Water: Total_Water = Rainfall_mm + Previous_Irrigation_mm
# ~Water_per_Hectare = (Rainfall_mm + Previous_Irrigation_mm) / Field_Area_hectare
# ~Evaporation = Temperature_C × Sunlight_Hours × cat__Wind_Speed_kmh
# ~Soil_Fertility = Organic_Carbon / (Electrical_Conductivity + 1)
# ~Mulch_Effect = cat__Mulching_Used × Soil_Moisture
# ~Heat_Stress = Temperature_C × ,Humidity

#Soil_pH,Soil_Moisture,Organic_Carbon,Electrical_Conductivity,
# Temperature_C,Humidity,Rainfall_mm,Sunlight_Hours,

# cat__Soil_Type,cat__Wind_Speed_kmh,cat__Crop_Type,cat__Crop_Growth_Stage,cat__Season,
# cat__Irrigation_Type,cat__Water_Source,cat__Field_Area_hectare,cat__Mulching_Used,
# cat__Soil_Moisture,cat__Region


import numpy as np
import pandas as pd
import os
import logging
from imblearn.over_sampling import SMOTENC
from collections import Counter
from sklearn.preprocessing import LabelEncoder

logging_path = r'C:\Irrigation_Water_Requirement\logging\egineering.log'
logging.basicConfig(
    filename=logging_path,
    filemode='a',
    level=logging.INFO,
    format= '%(asctime)s-%(levelname)s-%(message)s'
)

class Engineerig():
    def __init__(self, X_train, X_test, y_train):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = np.ravel(y_train)

    def feature_adding(self):
        try:
            #Total water
            self.X_train['Total_Water'] = self.X_train.Rainfall_mm + self.X_train.Soil_Moisture
            self.X_test['Total_Water'] = self.X_test.Rainfall_mm + self.X_test.Soil_Moisture

            # Evaporation
            self.X_train['Evaporation'] = self.X_train.Temperature_C * self.X_train.Sunlight_Hours
            self.X_test['Evaporation'] = self.X_test.Temperature_C * self.X_test.Sunlight_Hours

            # Soil_Fertility
            self.X_train['Soil_Fertility'] = self.X_train.Organic_Carbon / (self.X_train.Electrical_Conductivity + 1)
            self.X_test['Soil_Fertility'] = self.X_test.Organic_Carbon / (self.X_test.Electrical_Conductivity + 1)
            
            # Heat_Stress
            self.X_train['Heat_Stress'] = self.X_train.Temperature_C * self.X_train.Humidity
            self.X_test['Heat_Stress'] = self.X_test.Temperature_C * self.X_test.Humidity

            logging.info(f'Feature adding has been conducted')
            return self
        except Exception as e:
            logging.error(f'Error while adding features: {e}')
            raise

    # Handling the 60% skewed features

    def handle_skewness(self):
        try:
            num_cols = self.X_train.select_dtypes(include = [np.number]).columns.to_list()
            skewness = self.X_train[num_cols].skew()
            skewed_features = skewness[abs(skewness) > 0.60].index.to_list()

            for col in skewed_features:
                if (self.X_train[col] >= 0).all():
                    self.X_train[col] = np.log1p(self.X_train[col])
                    self.X_test[col] = np.log1p(self.X_test[col])
            logging.info(f'Skewness is handeled, Skewed Featrures: {skewed_features}')
            return self
        except Exception as e:
            logging.error(f'Error while transforming skewed features: {e}')
            raise
    def imbalance_handling(self):
        try:

            cat_cols = self.X_train.select_dtypes(exclude = [np.number]).columns.to_list()
            cat_indices = [self.X_train.columns.get_loc(col) for col in cat_cols]

            smote_nc = SMOTENC(categorical_features = cat_indices, random_state=42, k_neighbors=1)

            self.X_train, self.y_train = smote_nc.fit_resample(self.X_train, self.y_train)
            logging.info(f'Oversampling is done to ONLY TRAIN SET:{Counter(self.y_train)}')
            return self
        except Exception as e:
            logging.error(f'Error while oversampling the Train Set')
            raise

    def data_saving(self):
        data_path = r'C:\Irrigation_Water_Requirement\Data\engineered_Data'

        try:
            os.makedirs(data_path, exist_ok= True)
            path = os.path.join(data_path, 'X_train.csv')
            self.X_train.to_csv(path, index = False)

            os.makedirs(data_path, exist_ok= True)
            path = os.path.join(data_path, 'X_test.csv')
            self.X_test.to_csv(path, index = False)

            os.makedirs(data_path, exist_ok= True)
            path = os.path.join(data_path, 'y_train.csv')
            self.y_train = pd.Series(self.y_train, name='Irrigation_Need')
            self.y_train.to_csv(path, index = False)

            logging.info(f'Engineered data is saved at {data_path}')
            return self
        except Exception as e:
            logging.error(f'Error while saving data {e}')
            raise