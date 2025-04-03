"""
Create a placeholder machine learning model for the prediction agent.
This script generates a simple random forest regressor that predicts
days until failure based on sensor data features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Create a directory for the model if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic training data
n_samples = 1000
n_features = 14  # Match the number of features in our feature engineering

# Create feature names
feature_names = [
    'avg_temperature', 'avg_vibration', 'avg_pressure', 'avg_current', 
    'avg_rpm', 'avg_afr', 'std_temperature', 'std_vibration', 'std_pressure',
    'temp_trend', 'vibration_trend', 'max_vibration', 'max_temperature',
    'high_vibration_count'
]

# Generate random feature data
X = np.random.rand(n_samples, n_features)

# Create synthetic target values (days until failure)
# Higher vibration and temperature should lead to shorter time to failure
y = np.random.randint(1, 60, n_samples)  # 1-60 days
# Make vibration and temperature inversely related to days to failure
y = y - X[:, 1] * 20  # vibration impact
y = y - X[:, 0] * 15  # temperature impact
y = np.clip(y, 1, 120)  # Ensure values are between 1 and 120 days

# Create and train a simple Random Forest model
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(X, y)

# Save the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'failure_prediction_model.pkl')
joblib.dump(model, model_path)

print(f"Placeholder model saved to {model_path}")
print(f"Feature importance:")
for feature, importance in zip(feature_names, model.feature_importances_):
    print(f"  {feature}: {importance:.4f}")
