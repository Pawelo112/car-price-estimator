# model_training.py

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib

# Load the encoded data (created by data_preprocessing.py)
data_encoded = pd.read_csv('car_data_encoded.csv')
print('Encoded data loaded')

# Split into features and target
X = data_encoded.drop('price', axis=1)
y = data_encoded['price']

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


# Model 1: Decision Tree Regressor
# Initialize a Decision Tree with a minimum of 11 samples per leaf and max depth of 16
tree_model = DecisionTreeRegressor(min_samples_leaf=11, max_depth=16, random_state=42)

# Train the Decision Tree
tree_model.fit(X_train, y_train)
# Predict on the test set
y_pred_tree = tree_model.predict(X_test)
# Compute MAE
mae_tree = mean_absolute_error(y_test, y_pred_tree)
print(f"\nDecision Tree MAE: {mae_tree:.2f} $")


# Model 2: Random Forest Regressor
# Initialize a Random Forest with 300 trees
rf_model = RandomForestRegressor(n_estimators=300, random_state=42)

# Train the Random Forest
rf_model.fit(X_train, y_train)
# Predict on the test set
y_pred_rf = rf_model.predict(X_test)
# Compute MAE
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"\nRandom Forest MAE: {mae_rf:.2f} $")


# Model 3: XGBoost Regressor
# Initialize XGBoost with 300 estimators, max depth 7, learning rate 0.1, subsample 0.7, colsample_bytree 0.8
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

# Train XGBoost
xgb_model.fit(X_train, y_train)
# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)
# Compute MAE
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"\nXGBoost MAE: {mae_xgb:.2f} $")

# Saving the best model
joblib.dump(xgb_model, 'xgb_model.pkl')
print("\nModel training completed. Saved 'xgb_model.pkl'.")
