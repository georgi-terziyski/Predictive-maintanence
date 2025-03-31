from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
from datetime import datetime
import joblib
import numpy as np

load_dotenv()

app = Flask(__name__)

# Load ML model (placeholder - replace with actual model loading)
try:
    MODEL = joblib.load('model.pkl')
    MODEL_LOADED = True
except:
    MODEL = None
    MODEL_LOADED = False

@app.route('/health')
def health_check():
    model_status = 'loaded' if MODEL_LOADED else 'unavailable'
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'model': model_status,
        'version': '1.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Prediction model not loaded'}), 503

    try:
        data = request.json
        # Validate input data
        required_fields = ['temperature', 'vibration', 'load']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Prepare features for prediction
        features = np.array([
            float(data['temperature']),
            float(data['vibration']),
            float(data['load'])
        ]).reshape(1, -1)

        # Make prediction
        prediction = MODEL.predict(features)[0]
        return jsonify({
            'prediction': float(prediction),
            'timestamp': datetime.now().isoformat()
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=int(os.getenv('PREDICTION_AGENT_PORT')))
