from flask import Flask, jsonify, request
import requests
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import json

load_dotenv()

app = Flask(__name__)

# Configuration
DATA_AGENT_URL = os.getenv('DATA_AGENT_URL', 'http://localhost:5001')
MODEL_PATH = os.getenv('MODEL_PATH', 'agents/models/failure_prediction_model.pkl')
PREDICTION_HORIZON_DAYS = int(os.getenv('PREDICTION_HORIZON_DAYS', 30))

# Load ML model (placeholder - replace with actual model loading)
try:
    MODEL = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {str(e)}")
    MODEL = None
    MODEL_LOADED = False

@app.route('/health')
def health_check():
    # Check data agent connectivity
    data_agent_status = 'unavailable'
    try:
        response = requests.get(f"{DATA_AGENT_URL}/health", timeout=2)
        if response.status_code == 200:
            data_agent_status = response.json().get('status', 'unknown')
    except:
        pass
        
    model_status = 'loaded' if MODEL_LOADED else 'unavailable'
    
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'model': model_status,
        'data_agent': data_agent_status,
        'version': '1.0'
    })

def fetch_machine_data(machine_id):
    """Fetch sensor data for the specified machine from the data agent"""
    try:
        response = requests.get(
            f"{DATA_AGENT_URL}/predict?machine_id={machine_id}",
            timeout=5
        )
        
        if response.status_code != 200:
            return None, f"Data agent returned status code {response.status_code}"
            
        return response.json(), None
        
    except Exception as e:
        return None, f"Error fetching data from data agent: {str(e)}"

def prepare_features(sensor_data):
    """
    Process raw sensor data into features for the ML model
    This is where we would do feature engineering based on the 4 days of data
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(sensor_data)
    
    # Example feature engineering (would be customized based on your model):
    features = {
        # Recent averages
        'avg_temperature': df['temperature'].mean(),
        'avg_vibration': df['vibration'].mean(),
        'avg_pressure': df['pressure'].mean(),
        'avg_current': df['current'].mean(),
        'avg_rpm': df['rpm'].mean(),
        'avg_afr': df['afr'].mean(),
        
        # Recent standard deviations (variability)
        'std_temperature': df['temperature'].std(),
        'std_vibration': df['vibration'].std(),
        'std_pressure': df['pressure'].std(),
        
        # Trends (linear regression slopes would be better)
        'temp_trend': df['temperature'].iloc[-1] - df['temperature'].iloc[0],
        'vibration_trend': df['vibration'].iloc[-1] - df['vibration'].iloc[0],
        
        # Min/Max values
        'max_vibration': df['vibration'].max(),
        'max_temperature': df['temperature'].max(),
        
        # Count of outlier readings (e.g., vibration > 6.0)
        'high_vibration_count': len(df[df['vibration'] > 6.0]),
    }
    
    # Convert to numpy array for model input (order must match model training)
    feature_names = [
        'avg_temperature', 'avg_vibration', 'avg_pressure', 'avg_current', 
        'avg_rpm', 'avg_afr', 'std_temperature', 'std_vibration', 'std_pressure',
        'temp_trend', 'vibration_trend', 'max_vibration', 'max_temperature',
        'high_vibration_count'
    ]
    
    return np.array([features[name] for name in feature_names]).reshape(1, -1)

def store_prediction(machine_id, days_to_failure, confidence, contributing_factors):
    """Store prediction results back to the database via data agent"""
    try:
        # Calculate predicted failure date
        predicted_failure_date = (datetime.now() + timedelta(days=days_to_failure)).isoformat()
        
        # Prepare prediction details
        prediction_data = {
            'machine_id': machine_id,
            'predicted_failure_date': predicted_failure_date,
            'confidence': confidence,
            'model_version': '1.0.0',
            'prediction_details': json.dumps({
                'days_to_failure': days_to_failure,
                'contributing_factors': contributing_factors,
                'prediction_date': datetime.now().isoformat()
            })
        }
        
        # Send to data agent
        response = requests.post(
            f"{DATA_AGENT_URL}/predictions",
            json=prediction_data,
            timeout=5
        )
        
        if response.status_code != 201:
            return False, f"Failed to store prediction: {response.text}"
            
        return True, response.json().get('id')
        
    except Exception as e:
        return False, f"Error storing prediction: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Prediction model not loaded'}), 503

    try:
        data = request.json
        
        # Now we just need the machine_id
        if 'machine_id' not in data:
            return jsonify({'error': 'Missing machine_id field'}), 400
            
        machine_id = data['machine_id']
        
        # Fetch machine data from data agent
        machine_data, error = fetch_machine_data(machine_id)
        if error:
            return jsonify({'error': error}), 500
            
        # Process sensor data into features
        sensor_readings = machine_data.get('sensor_data', [])
        if not sensor_readings:
            return jsonify({'error': 'No sensor data available for this machine'}), 404
            
        # Prepare features for the model
        features = prepare_features(sensor_readings)
        
        # Make prediction
        # For this example, we'll assume the model predicts days until failure
        days_to_failure = max(0, float(MODEL.predict(features)[0]))
        
        # Calculate confidence (this would normally come from the model)
        confidence = 0.85  # Placeholder
        
        # Identify contributing factors (this could be from feature importance)
        contributing_factors = ["high vibration", "temperature fluctuation"]
        
        # Store prediction in database
        success, result = store_prediction(
            machine_id, 
            days_to_failure, 
            confidence, 
            contributing_factors
        )
        
        if not success:
            print(f"Warning: Failed to store prediction: {result}")
        
        # Return prediction to caller
        return jsonify({
            'machine_id': machine_id,
            'predicted_failure_date': (datetime.now() + timedelta(days=days_to_failure)).isoformat(),
            'days_to_failure': days_to_failure,
            'confidence': confidence,
            'contributing_factors': contributing_factors,
            'prediction_id': result if success and isinstance(result, int) else None,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=int(os.getenv('PREDICTION_AGENT_PORT', 5002)))
