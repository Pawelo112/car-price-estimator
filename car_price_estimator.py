# app.py

# Import libraries
from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np


app = Flask(__name__)

# Load the best model and helper files
xgb_model = joblib.load('xgb_model.pkl')
le_encoders = joblib.load('le_encoders.pkl')
top_models = joblib.load('top_models.pkl')
feature_names = joblib.load('feature_names.pkl')


def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function to prepare form data for the XGBoost model.
    """
    # Convert 'levy' (replace '-' with NaN, then fill with 0)
    data['levy'] = data['levy'].replace('-', np.nan).astype(float).fillna(0)

    # Convert 'engine volume' (remove ' Turbo' and replace comma with dot, then float)
    data['engine volume'] = (
        data['engine volume']
        .str.replace(' Turbo', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

    # Convert 'mileage' to integer
    data['mileage'] = data['mileage'].astype(int)

    # Convert 'prod. year', 'cylinders', 'airbags' to integers
    data['prod. year'] = data['prod. year'].astype(int)
    data['cylinders'] = data['cylinders'].astype(int)
    data['airbags'] = data['airbags'].astype(int)

    # Binary encoding for 'leather interior', 'wheel', and 'doors' using saved LabelEncoders
    for col, le in le_encoders.items():
        data[col] = le.transform(data[col])

    # For 'model', replace anything not in top_models with 'Other' and then one‐hot encode
    data['model'] = data['model'].apply(lambda x: x if x in top_models else 'Other')
    data = pd.get_dummies(data, columns=['model'], prefix='model')

    # One‐hot encode the remaining categorical features
    cat_cols = [
        'category',
        'fuel type',
        'gear box type',
        'drive wheels',
        'manufacturer',
        'color'
    ]
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    # Add any missing columns (fill with zeros) so that the DataFrame matches feature_names exactly
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0
    data = data[feature_names]

    return data


@app.route('/', methods=['GET'])
def form():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data into a dictionary
    user_input = {
        'levy': request.form['levy'],
        'manufacturer': request.form['manufacturer'],
        'model': request.form['model'],
        'prod. year': request.form['prod_year'],
        'category': request.form['category'],
        'leather interior': request.form['leather_interior'],
        'fuel type': request.form['fuel_type'],
        'engine volume': request.form['engine_volume'],
        'mileage': request.form['mileage'],
        'cylinders': request.form['cylinders'],
        'gear box type': request.form['gear_box_type'],
        'drive wheels': request.form['drive_wheels'],
        'doors': request.form['doors'],
        'wheel': request.form['wheel'],
        'color': request.form['color'],
        'airbags': request.form['airbags']
    }
    data = pd.DataFrame(user_input, index=[0])
    x = preprocess_input(data)

    # Model prediction
    pred = round(xgb_model.predict(x)[0], 2)
    return render_template('result.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)
