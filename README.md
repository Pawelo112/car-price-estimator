# 🚗Car Price Prediction

## 📚 Table of Contents
<details>
<summary><strong>Click to expand</strong></summary>

- [📖 Dataset](#-dataset)  
- [📋 Project Structure](#-project-structure)  
- [📝 Design Decisions & Implementation Details](#-design-decisions--implementation-details)  
  - [📊 Data Preparation](#-data-preparation)  
  - [🔧 Feature Engineering & Encoding](#-feature-engineering--encoding)  
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

## 📖License
Copyright © 2025 [Paweł Marcinkowski](https://github.com/Pawelo112), [Wiktor Błaszkiewicz](https://github.com/qub1itz).  
This project is [MIT](https://github.com/Pawelo112/car-price-estimator/blob/main/LICENSE) licensed.
