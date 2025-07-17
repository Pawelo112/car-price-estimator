# 🚗Car Price Prediction

## 📚 Table of Contents
<details>
<summary><strong>Click to expand</strong></summary>

- [🔍 Overview](#-overview)
- [📖 Dataset](#-dataset)  
- [📋 Project Structure](#-project-structure)  
- [📝 Design Decisions & Implementation Details](#-design-decisions--implementation-details)  
  - [📊 Data Preparation](#-data-preparation)  
  - [🔧 Feature Engineering & Encoding](#-feature-engineering--encoding)  
- [📊 Visualization](#-visualization)
- [📈 Model Training](#-model-training) 
- [🖥️ Flask Web Application](#-flask-web-application)
- [🧰 Tools](#-tools)  
- [👨‍🏫 How to Run](#-how-to-run)  
- [📌 Conclusion](#-conclusion)  
- [📖 License](#license)  

</details>

## 🔍 Overview

This project is a complete pipeline for predicting used car prices based on various vehicle attributes.  

The project includes:
- Data preparation & cleaning  
- Feature engineering & encoding  
- Training multiple regression models and selecting best one 
- Visual exploratory data analysis (EDA)  
- A deployed Flask web app for real-time predictions

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
├── templates/  
  &ensp; ├── index.html # Input form  
  &ensp; └── result.html # Display prediction  
├── requirements.txt # Python dependencies  
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

### Example usage



## 🧰 Tools
 
- **Core Libraries:**  
  - `pandas`, `numpy`  
  - `scikit-learn`  
  - `xgboost`  
  - `joblib`  
  - `flask`  
  - `seaborn`, `matplotlib`, `ydata_profiling` 

## 👨‍🏫 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction
```

### 2. Set Up Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge).  
Place car_price_prediction.csv in the root folder.  

### 5. Preprocess Data

```bash
python data_preprocessing.py
```

### 6. Train Models

```bash
python model_training.py
```  

### 7. Run Flask App

```bash
python app.py
```
Visit http://127.0.0.1:5000 in your browser and try making some predictions.  

## 📌 Conclusion
This project demonstrates an end-to-end machine learning workflow for predicting car prices, from raw data to a functional web app.  
You are more than welcome to try it out yourself and make any changes you want.

## 📖License
Copyright © 2025 [Paweł Marcinkowski](https://github.com/Pawelo112), [Wiktor Błaszkiewicz](https://github.com/qub1itz).  
This project is [MIT](https://github.com/Pawelo112/car-price-estimator/blob/main/LICENSE) licensed.
