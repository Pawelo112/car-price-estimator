# 🚗Car Price Prediction

## 📚 Table of Contents
<details>
<summary><strong>Click to expand</strong></summary>

- [📖 Dataset](#-dataset)  
- [📋 Project Structure](#-project-structure)  
- [📝 Design Decisions & Implementation Details](#-design-decisions--implementation-details)  
  - [📊 Data Preparation](#-data-preparation)  
  - [🔧 Feature Engineering & Encoding](#-feature-engineering--encoding)  
- [📊 Visualization](#-visualization)
- [📈 Model Training](#-model-training) 
- [🖥️ Flask Web Application](#-flask-web-application)   
- [📖 License](#license)  

</details>

## 📖 Dataset

- **Source:** [Kaggle – Car Price Prediction Challenge](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge)  
- **Rows/Columns:** 19,237 × 18  
- **Target:** Price (USD)  
- **Features include:**  
  Manufacturer, model, year, mileage, engine volume, fuel type, gear box, color, doors, etc.

## 📋 Project Structure

car_price_prediction/  
│  
├── data_preprocessing.py # Clean & encode dataset  
├── car_price_prediction.csv # Raw Kaggle dataset  
├── visualization.ipynb # EDA notebook with profiling & plots  
├── model_training.py # Train regressors & evaluate performance  
├── car_price_estimator.py # Flask server app for predictions
└── README.md # This file  

## 📝 Design Decisions & Implementation Details

### 📊 Data Preparation

- Drop unnecessary columns (e.g., `ID`)
- Convert units (e.g., `mileage`, `levy`, `engine volume`)
- Remove duplicates & outliers (e.g., mileage > 1M km)
- Ensure no missing values remain

### 🔧 Feature Engineering & Encoding

- **Binary Columns:** Label encoding for `leather interior`, `wheel`, `doors`  
- **High Cardinality Columns:** Keep only top 20 frequent models  
- **One-Hot Encoding:** For categorical fields (e.g., model, manufacturer)  
- **Artifacts Saved:**
  - `car_price_prediction_cleaned.csv`
  - `car_data_encoded.csv`  
  - `le_encoders.pkl`  
  - `top_models.pkl`  
  - `feature_names.pkl`  

## 📊 Visualization

Found in visualization.ipynb, includes:

- Automatic profiling with ydata_profiling  
- Histograms: Price, year, engine volume, etc.  
- Correlation heatmaps  
- Frequency plots by category, model, color

## 📈 Model Training

- **Split:** 80/20 train-test  
- **Models Trained:**  
  - Decision Tree (`min_samples_leaf=11`, `max_depth=16`)  
  - Random Forest (`n_estimators=300`)  
  - XGBoost (`n_estimators=300`, `max_depth=7`, `learning_rate=0.1`, `subsample=0.7`,
    `colsample_bytree=0.8`)  
- **Evaluation Metric:** Mean Absolute Error (MAE)  
- **Best Model Saved:** `xgb_model.pkl`  

## 💻 Flask Web Application

### Routes

- / → Input form (index.html)  
- /predict → Output price prediction (result.html)  

### Features

- Accepts full car attributes
- Applies saved preprocessing steps (from training)
- Displays formatted price prediction (e.g., $12,345.67)
- Option to return to form and predict again

## 📖License
Copyright © 2025 [Paweł Marcinkowski](https://github.com/Pawelo112), [Wiktor Błaszkiewicz](https://github.com/qub1itz).  
This project is [MIT](https://github.com/Pawelo112/car-price-estimator/blob/main/LICENSE) licensed.
