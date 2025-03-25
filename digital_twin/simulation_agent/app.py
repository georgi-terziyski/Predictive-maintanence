from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
from datetime import datetime
import numpy as np
import random

load_dotenv()

app = Flask(__name__)

# Simulation parameters (can be loaded from config/database)
SIMULATION_PARAMS = {
    'temperature_range': (20, 100),
    'vibration_range': (0, 10),
    'load_range': (0, 100),
    'default_duration': 24  # hours
}

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    })

@app.route('/simulate', methods=['POST'])
def run_simulation():
    try:
        params = request.json
        
        # Validate input parameters
        if not params:
            params = {}
            
        # Set simulation parameters with defaults
        temp_min, temp_max = SIMULATION_PARAMS['temperature_range']
        vib_min, vib_max = SIMULATION_PARAMS['vibration_range']
        load_min, load_max = SIMULATION_PARAMS['load_range']
        
        duration = int(params.get('duration', SIMULATION_PARAMS['default_duration']))
        steps = duration * 2  # 30-minute intervals
        
        # Generate simulated time series data
        timestamps = []
        temperatures = []
        vibrations = []
        loads = []
        
        for i in range(steps):
            # Generate realistic trends with some randomness
            timestamps.append(datetime.now().isoformat())
            
            # Base values with some drift and noise
            temp_base = random.uniform(temp_min, temp_max)
            temp_drift = np.sin(i/steps * np.pi) * (temp_max - temp_min) * 0.2
            temperatures.append(temp_base + temp_drift + random.uniform(-5, 5))
            
            vib_base = random.uniform(vib_min, vib_max)
            vibrations.append(vib_base + random.uniform(-1, 1))
            
            load_base = random.uniform(load_min, load_max)
            loads.append(load_base + random.uniform(-10, 10))
            
        return jsonify({
            'simulation_id': f'sim_{datetime.now().strftime("%Y%m%d%H%M%S")}',
            'parameters': {
                'duration_hours': duration,
                'temperature_range': [temp_min, temp_max],
                'vibration_range': [vib_min, vib_max],
                'load_range': [load_min, load_max]
            },
            'data': {
                'timestamps': timestamps,
                'temperatures': temperatures,
                'vibrations': vibrations,
                'loads': loads
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=int(os.getenv('SIMULATION_AGENT_PORT')))
