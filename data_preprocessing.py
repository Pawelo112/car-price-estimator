# data_preprocessing.py

# Importing libraries

# Basic
import pandas as pd
import numpy as np
# For saving encoders and model artifacts
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('car_price_prediction.csv')
print('Dataset loaded.')

# Drop the 'ID' column (unnecessary for further analysis)
data = data.drop(columns='ID')

# Rename all columns to lowercase for consistency
data.columns = data.columns.str.lower()

# Convert 'mileage' from string with " km" to integer
data['mileage'] = data['mileage'].str.replace(' km', '', regex=True).astype('int64')

# Checking for duplicates and remove them to avoid artificially inflating certain price values
print(f'\nNumber of duplicates in the dataset: {data.duplicated().sum()}')
data = data.drop_duplicates()

print(f'\nNumber of duplicates after removal: {data.duplicated().sum()}')
print(f'New shape after dropping duplicates: {data.shape}')

# Checking for any missing values
print('\nMissing values in each column:')
print(data.isna().sum())

# Removing outliers based on price:
# Compute the 99th percentile price (only 1% of data is above this value)
upper_price_limit = data['price'].quantile(0.99)
print(f"\n99th percentile of price: {upper_price_limit:.2f} $")

# Remove any record with price > 99th percentile
data_cleaned = data[data.price <= upper_price_limit]
print(f"Shape after removing upper‐price outliers: {data_cleaned.shape}")

# Compute the 5th percentile price (only 5% of data is below this)
lower_price_limit = data.price.quantile(0.05)
print(f"\n5th percentile of price: {lower_price_limit:.2f} $")

# Remove any record with price ≤ 5th percentile
data_cleaned = data_cleaned[data_cleaned.price > lower_price_limit]
print(f"Shape after removing lower‐price outliers: {data_cleaned.shape}")

# Remove unrealistic mileage values (> 1,000,000 km)
upper_mileage_limit = data.mileage.quantile(0.99)
print(f"\n99th percentile of mileage: {upper_mileage_limit:.2f} km")
print(f'Number of cars >1,000,000 km: {(data.mileage > 1_000_000).sum()}')

data_cleaned = data_cleaned[data_cleaned.mileage < 1_000_000]
print(f"Shape after removing unrealistic mileage: {data_cleaned.shape}")

# Saving data after cleaning to csv file for visualization (included in visualization.ipynb)
data_cleaned.to_csv('car_price_prediction_cleaned.csv', index=False)

# Convert 'levy' from string to numeric, replacing '-' with NaN then filling with 0
data_cleaned['levy'] = data_cleaned['levy'].replace('-', np.nan)
data_cleaned['levy'] = pd.to_numeric(data_cleaned['levy'], errors='coerce').fillna(0)

# Convert 'engine volume' from strings like "2.0 Turbo" to float, removing “ Turbo” and replacing comma with dot
data_cleaned['engine volume'] = (
    data_cleaned['engine volume']
    .str.replace(' Turbo', '', regex=False)
    .str.replace(',', '.', regex=False)
)
data_cleaned['engine volume'] = pd.to_numeric(data_cleaned['engine volume'], errors='coerce')

# Binary encoding for 'leather interior', 'wheel', and 'doors' using LabelEncoder
binary_columns = ['leather interior', 'wheel', 'doors']
le_encoders = {}
for col in binary_columns:
    le = LabelEncoder().fit(data_cleaned[col])
    le_encoders[col] = le
    data_cleaned[col] = le.transform(data_cleaned[col])

# Save binary encoders for later use in the Flask app
joblib.dump(le_encoders, 'le_encoders.pkl')

# Identify the top 20 most frequent models
top_models = data_cleaned['model'].value_counts().nlargest(20).index.tolist()
# For any model not in the top 20, label it as 'Other'
data_cleaned['model'] = data_cleaned['model'].apply(lambda x: x if x in top_models else 'Other')
# Save the top_models list for the Flask app
joblib.dump(top_models, 'top_models.pkl')

# One‐Hot encode 'model' (now with 21 categories: top 20 + 'Other')
# and one‐hot encode these categorical columns with drop_first=True:
# ['category', 'fuel type', 'gear box type', 'drive wheels', 'manufacturer', 'color']
data_encoded = pd.get_dummies(
    data_cleaned,
    columns=[
        'model',
        'category',
        'fuel type',
        'gear box type',
        'drive wheels',
        'manufacturer',
        'color'
    ],
    drop_first=True
)

# Verify no text columns remain (all are numeric)
text_columns = data_encoded.select_dtypes(include=['object']).columns.tolist()
print(f"\nText columns remaining after encoding (should be []): {text_columns}")

# Verify that there are no missing values in 'engine volume'
print(f"\nMissing values in 'engine volume': {data_encoded['engine volume'].isna().sum()}")

# Save feature names (all columns except 'price') for later ordering in the Flask app
feature_names = data_encoded.drop('price', axis=1).columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# Split into X and y (features and target)
X = data_encoded.drop('price', axis=1)
y = data_encoded['price']

# Saving the cleaned and encoded DataFrame to csv
data_encoded.to_csv('car_data_encoded.csv', index=False)

# End of data_preprocessing.py
if __name__ == '__main__':
    print("\nData preprocessing completed. Files saved:\n"
          "- car_price_prediction_cleaned.csv\n"
          "- le_encoders.pkl\n"
          "- top_models.pkl\n"
          "- feature_names.pkl\n"
          "- car_data_encoded.csv")
