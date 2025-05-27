from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timedelta
import numpy as np
import random
import json

load_dotenv()

app = Flask(__name__)

# Configuration
PREDICTION_AGENT_URL = os.getenv('PREDICTION_AGENT_URL', 'http://localhost:5002')

# Default sensor ranges
SENSOR_DEFAULTS = {
    'temperature': {'default': 75.0, 'variation': 0.5},
    'pressure': {'default': 4.5, 'variation': 0.1},
    'vibration': {'default': 2.5, 'variation': 0.2},
    'current': {'default': 28.0, 'variation': 0.3},
    'rpm': {'default': 3200, 'variation': 20},
    'afr': {'default': 13.0, 'variation': 0.05}
}

# Configuration
DATA_AGENT_URL = os.getenv('DATA_AGENT_URL', 'http://localhost:5001')
PREDICTION_AGENT_URL = os.getenv('PREDICTION_AGENT_URL', 'http://localhost:5002')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    # Check prediction and data agent connectivity
    prediction_agent_status = 'unavailable'
    data_agent_status = 'unavailable'
    
    try:
        response = requests.get(f"{PREDICTION_AGENT_URL}/health", timeout=2)
        if response.status_code == 200:
            prediction_agent_status = response.json().get('status', 'unknown')
    except:
        pass
        
    try:
        response = requests.get(f"{DATA_AGENT_URL}/health", timeout=2)
        if response.status_code == 200:
            data_agent_status = response.json().get('status', 'unknown')
    except:
        pass
    
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'prediction_agent': prediction_agent_status,
        'data_agent': data_agent_status,
        'version': '1.0'
    })

def generate_time_series(machine_id, initial_values=None, fixed_parameters=None, duration_hours=24, interval_minutes=30):
    """
    Generate a realistic time series of sensor data based on initial values and fixed parameters.
    
    Args:
        machine_id: Machine identifier
        initial_values: Dictionary of initial sensor values
        fixed_parameters: Dictionary of parameters to keep constant
        duration_hours: Duration of simulation in hours
        interval_minutes: Time interval between data points in minutes
    
    Returns:
        List of dictionaries containing the time series data
    """
    if initial_values is None:
        initial_values = {}
    
    if fixed_parameters is None:
        fixed_parameters = {}
    
    # Calculate number of intervals
    intervals = int((duration_hours * 60) / interval_minutes)
    
    # Initialize the time series
    time_series = []
    start_time = datetime.now()
    
    # Set initial values from defaults or provided values
    current_values = {}
    for sensor, config in SENSOR_DEFAULTS.items():
        if sensor in initial_values:
            current_values[sensor] = initial_values[sensor]
        else:
            current_values[sensor] = config['default']
    
    # Override with fixed parameters if specified
    for param, value in fixed_parameters.items():
        if param in SENSOR_DEFAULTS:
            current_values[param] = value
    
    # Generate time series with realistic trends
    for i in range(intervals):
        timestamp = start_time + timedelta(minutes=i*interval_minutes)
        
        # Create data point
        data_point = {
            'timestamp': timestamp.isoformat(),
            'machine_id': machine_id
        }
        
        # Update sensor values with realistic variations
        for sensor, config in SENSOR_DEFAULTS.items():
            if sensor in fixed_parameters:
                # Use fixed value
                data_point[sensor] = fixed_parameters[sensor]
            else:
                # Apply random variation
                variation = config['variation']
                new_value = current_values[sensor] + random.uniform(-variation, variation)
                
                # Update current value for next iteration (cumulative effect)
                current_values[sensor] = new_value
                
                # Round appropriately and add to data point
                if sensor == 'rpm':
                    data_point[sensor] = int(new_value)
                else:
                    data_point[sensor] = round(new_value, 2)
        
        time_series.append(data_point)
    
    return time_series

def store_simulation(machine_id, parameters, simulated_data, prediction_result):
    """
    Store simulation results to the database via data agent.
    
    Args:
        machine_id: Machine identifier
        parameters: Simulation parameters
        simulated_data: Generated time series data
        prediction_result: Prediction results
        
    Returns:
        (success, simulation_id or error_message)
    """
    try:
        # Create a meaningful scenario type from the fixed parameters
        if 'fixed_parameters' in parameters and parameters['fixed_parameters']:
            # Use the fixed parameters as the scenario type
            scenario_parts = []
            for param, value in parameters['fixed_parameters'].items():
                scenario_parts.append(f"{param}={value}")
            scenario_type = " & ".join(scenario_parts)
        else:
            scenario_type = "baseline_simulation"
            
        # Create enhanced parameters object that includes the generated data
        enhanced_parameters = parameters.copy()  # Copy the original parameters
        enhanced_parameters['generated_dataset'] = simulated_data  # Add the full dataset
        
        # Format the simulation data for storage
        simulation_data = {
            'machine_id': machine_id,
            'scenario_type': scenario_type[:50],  # Ensure it fits in the field
            'parameters': enhanced_parameters,  # Use enhanced parameters with dataset
            'results': {
                'prediction': prediction_result,
                'data_summary': {
                    'data_points': len(simulated_data),
                    'start_time': simulated_data[0]['timestamp'],
                    'end_time': simulated_data[-1]['timestamp']
                }
            }
        }
        
        # Send to data agent
        response = requests.post(
            f"{DATA_AGENT_URL}/simulations",
            json=simulation_data,
            timeout=5
        )
        
        if response.status_code != 201:
            return False, f"Failed to store simulation: {response.text}"
            
        return True, response.json().get('id')
        
    except Exception as e:
        return False, f"Error storing simulation: {str(e)}"

def get_prediction(machine_id, simulated_data):
    """
    Send simulated data to prediction agent to get prediction.
    
    Args:
        machine_id: Machine identifier
        simulated_data: List of simulated data points
    
    Returns:
        Prediction result or error message
    """
    try:
        # Format the data for the prediction agent
        # The prediction agent expects a list of sensor readings in the 'sensor_data' field
        formatted_data = {
            'machine_id': machine_id,
            'simulation_data': simulated_data,  # Send the full simulated dataset
            'simulation_mode': True  # Explicitly mark this as a simulation to skip DB storage
        }
        
        # Send the data to the prediction agent
        response = requests.post(
            f"{PREDICTION_AGENT_URL}/predict",
            json=formatted_data,
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f'Prediction failed: {response.status_code} - {response.text}'}
            
    except Exception as e:
        return {'error': f'Error calling prediction agent: {str(e)}'}

@app.route('/simulate', methods=['POST'])
def run_simulation():
    """
    Run a simulation based on provided parameters and get a prediction.
    
    Expected JSON input:
    {
        "machine_id": "M001",
        "initial_values": {
            "temperature": 75.0,
            "pressure": 4.5,
            ...
        },
        "fixed_parameters": {
            "temperature": 100.0  // Keep temperature fixed at 100°C
        },
        "duration_hours": 24
    }
    """
    try:
        # Get and validate parameters
        params = request.get_json() or {}
        
        # Check required parameters
        machine_id = params.json.get("machine_id", "") or params.get("machine_id", "")
        if not machine_id:
            return jsonify({'error': 'machine_id is required'}), 400
        
        # Get optional parameters
        initial_values = params.get('initial_values', {})
        fixed_parameters = params.get('fixed_parameters', {})
        duration_hours = int(params.get('duration_hours', 24))
        interval_minutes = int(params.get('interval_minutes', 30))
        
        # Generate the time series
        simulated_data = generate_time_series(
            machine_id,
            initial_values=initial_values,
            fixed_parameters=fixed_parameters,
            duration_hours=duration_hours,
            interval_minutes=interval_minutes
        )
        
        # Get prediction based on simulated data
        prediction_result = get_prediction(machine_id, simulated_data)
        
        # Create parameters object
        simulation_params = {
            'duration_hours': duration_hours,
            'interval_minutes': interval_minutes,
            'initial_values': initial_values,
            'fixed_parameters': fixed_parameters
        }
        
        # Store simulation results in database
        success, simulation_id = store_simulation(
            machine_id, 
            simulation_params, 
            simulated_data, 
            prediction_result
        )
        
        if not success:
            print(f"Warning: Failed to store simulation: {simulation_id}")
            # Continue even if storage fails - we'll still return the results
            simulation_id = f"sim_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Return only the prediction results, not the parameters or simulated data
        return jsonify({
            'simulation_id': simulation_id if isinstance(simulation_id, (int, str)) else f"sim_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'machine_id': machine_id,
            'prediction': prediction_result,
            'stored_in_database': success
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('SIMULATION_AGENT_PORT', 5003))
    app.run(host='0.0.0.0', port=port)
