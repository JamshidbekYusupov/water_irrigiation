# 🌱 Smart Irrigation Water Requirement Prediction

## 📌 Overview

The **Smart Irrigation Water Requirement Prediction** project focuses on building intelligent machine learning models to optimize irrigation usage based on environmental and agricultural conditions.

Efficient irrigation is critical for:

* 🌍 Sustainable agriculture
* 💧 Water conservation
* 🌾 Crop yield optimization

---

## 📂 Project Structure

```
├── Data/
│   ├── Raw_Data/
│   ├── Feature_analysis/
│   ├── Metrics/
│   └── logging/
│
├── Models/
│   └── all_models/
│
├── Notebooks/
│   ├── EDA.ipynb
│   └── model_building.ipynb
│
├── Scripts/
│   ├── baseline_model.py
│   ├── data_imputer.py
│   ├── feature_engineering.py
│   └── tuning_pipeline.py
│
├── src/
│   ├── baseline.py
│   ├── data_preprocess.py
│   ├── feature_engineering.py
│   └── tuning_pipeline.py
│
├── xAI/
│   └── SHAP.ipynb
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

The dataset includes key agronomic and environmental features:

* 🌱 Crop Type
* 🌍 Soil Type
* 🌡 Temperature
* 🌧 Rainfall
* 💨 Wind Speed
* ☀️ Sunlight Hours
* 💧 Soil Moisture

---

## 🎯 Objective

To build machine learning models that:

* Predict irrigation requirements
* Reduce water waste
* Improve agricultural efficiency

---

## ⚙️ Feature Engineering

👉 Click here to view implementation:
➡️ [Feature Engineering Code](./src/feature_engineering.py)

### 🔥 Newly Engineered Features

| Feature            | Formula                                        | Purpose               |
| ------------------ | ---------------------------------------------- | --------------------- |
| **Total_Water**    | Rainfall + Soil_Moisture                       | Total available water |
| **Evaporation**    | Temperature × Sunlight_Hours                   | Water loss estimation |
| **Soil_Fertility** | Organic_Carbon / (Electrical_Conductivity + 1) | Soil health           |
| **Heat_Stress**    | Temperature × Humidity                         | Crop stress level     |

---

### 📉 Skewness Handling

* Applied **log transformation** for skewed features (> 0.6)
* Ensures better model performance

---

### ⚖️ Imbalance Handling

* Used **SMOTENC** for mixed categorical + numerical data
* Applied only on training data

---

## 🧠 Machine Learning Models

### Baseline Models

* Decision Tree 🌳
* Stacking Classifier ⚡

### Advanced Pipeline

* Preprocessing (Scaling + Encoding)
* Feature Engineering
* Hyperparameter Tuning (GridSearchCV)

---

## 🔧 Pipeline Architecture

```
Raw Data → Imputation → Feature Engineering → Skew Handling → SMOTENC →
ColumnTransformer → Model → Evaluation
```

---

## 📈 Evaluation Metrics

* Accuracy
* Precision (Weighted)
* Recall (Weighted)
* F1 Score (Weighted)

Results are stored in:
📁 `Data/Metrics/`

---

## 🔍 Explainability

Model interpretability using SHAP:

➡️ [SHAP Analysis](./xAI/SHAP.ipynb)

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
```

```bash
python Scripts/baseline_model.py
```

```bash
python Scripts/tuning_pipeline.py
```

---

## 📌 Key Highlights

✔ End-to-end ML pipeline
✔ Feature engineering based on domain knowledge
✔ Handles skewness & imbalance
✔ Modular and scalable structure
✔ Explainable AI integration

---

## 📬 Future Improvements

* Add deep learning models
* Deploy as web application
* Real-time irrigation prediction system

---

## 👨‍💻 Author

**Jamusan**
Master’s in Business IT | ML & AI Enthusiast

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share!
