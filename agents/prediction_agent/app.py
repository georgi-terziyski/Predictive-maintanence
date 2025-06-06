from flask import Flask, jsonify, request
import requests
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import pandas as pd
import json
import tempfile
import subprocess
import re
import sys

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_validation_tool import standardize_and_validate_csv

load_dotenv()

app = Flask(__name__)

# Configuration
DATA_AGENT_URL = os.getenv('DATA_AGENT_URL', 'http://localhost:5001')
EQUIPMENT_FILE = os.getenv('EQUIPMENT_FILE', 'inference/data/10apr/equipment_usage.csv')
FAILURE_FILE = os.getenv('FAILURE_FILE', 'inference/data/10apr/failure_logs.csv')
MAINTENANCE_FILE = os.getenv('MAINTENANCE_FILE', 'inference/data/10apr/maintenance_history.csv')
PREDICTION_HORIZON_DAYS = int(os.getenv('PREDICTION_HORIZON_DAYS', 30))

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
    
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'data_agent': data_agent_status,
        'version': '1.0'
    })

def fetch_machine_data(machine_id):
    """Fetch prediction data for the specified machine from the data agent"""
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

def convert_to_inference_format(prediction_data, machine_id):
    """
    Convert API prediction data to the standardized format expected by inference.py.
    This function ensures that the column names and order match the training data.
    """
    # Create a DataFrame from the prediction readings
    df = pd.DataFrame(prediction_data)

    # Define the full mapping from source (lowercase) to target (PascalCase/UPPERCASE)
    column_mapping = {
        'timestamp': 'Timestamp',
        'machine_id': 'Machine_ID',
        'afr': 'AFR',
        'current': 'Current',
        'pressure': 'Pressure',
        'rpm': 'RPM',
        'temperature': 'Temperature',
        'vibration': 'Vibration'
    }

    # Rename columns based on the mapping
    df.rename(columns=column_mapping, inplace=True, errors='ignore')

    # If Machine_ID column was not in the original data (e.g., from simulation), add it
    if 'Machine_ID' not in df.columns:
        df['Machine_ID'] = machine_id

    # Define the required columns in the correct order to match the training data format
    required_columns = [
        'Timestamp', 'Machine_ID', 'AFR', 'Current', 'Pressure', 'RPM', 'Temperature', 'Vibration'
    ]

    # Ensure all required columns are present, fill with NaN if not.
    # This handles cases where the source data might be missing a column.
    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA
            
    # Select and reorder columns to ensure a consistent format
    df = df[required_columns]
    
    # Ensure timestamp is datetime and timezone-unaware
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', errors='coerce').dt.tz_localize(None)
    
    # Write to temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as temp_file:
        df.to_csv(temp_file.name, index=False)
        return temp_file.name

def store_prediction(machine_id, result):
    """Store prediction results back to the database via data agent"""
    try:
        # Extract relevant information from the inference result as per DB schema requirements
        # Send the result object directly as prediction_details - data_agent will handle JSON conversion
        prediction_data = {
            'machine_id': machine_id,
            'status': result['status'],
            'failure_probability': result['stage1_probability'],
            'prediction_timestamp': result['timestamp'],
            'prediction_details': result  # Send the entire result object, not as a string
        }
        
        print(f"Storing prediction for machine {machine_id} with status {result['status']}")
        
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
    try:
        data = request.json
        
        # Check for required field
        if 'machine_id' not in data:
            return jsonify({'error': 'Missing machine_id field'}), 400
            
        machine_id = data['machine_id']
        
        # Check if this is a simulation request
        is_simulation = 'simulation_data' in data or data.get('simulation_mode', False)
        
        if is_simulation:
            # Use the provided simulation data directly
            print(f"Using simulation data for machine ID: {machine_id}")
            prediction_readings = data['simulation_data']
        else:
            # Fetch machine data from data agent
            machine_data, error = fetch_machine_data(machine_id)
            if error:
                return jsonify({'error': error}), 500
                
            # Process prediction data
            prediction_readings = machine_data.get('prediction_data', [])
            if not prediction_readings:
                print(f"No prediction data available for machine ID: {machine_id}, Response: {machine_data}")
                return jsonify({'error': f'No prediction data available for machine ID: {machine_id}. Try one of these IDs: M001 through M010'}), 404
        
        # Convert to CSV for inference.py
        temp_csv_path = convert_to_inference_format(prediction_readings, machine_id)
        
        # Standardize the temporary CSV file before running inference
        try:
            print(f"Standardizing CSV file: {temp_csv_path}")
            standardize_and_validate_csv(temp_csv_path)
            print("CSV standardization successful.")
        except Exception as e:
            print(f"Error during CSV standardization: {e}")
            # Optionally, return an error response if standardization is critical
            return jsonify({'error': f'Failed to standardize data: {e}'}), 500
        
        # Save the standardized CSV to the logs directory
        log_filename = f"inference_input_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_path = os.path.join('logs', log_filename)
        os.makedirs('logs', exist_ok=True)
        import shutil
        shutil.copy(temp_csv_path, log_path)
        print(f"Saved standardized CSV to {log_path}")

        # Current timestamp
        current_timestamp = datetime.now().isoformat()
        
        # Execute inference.py as a subprocess
        process = subprocess.Popen([
            'python', 'inference/inference.py',
            '--current_timestamp', current_timestamp,
            '--input_data', temp_csv_path,
            '--machine_id', machine_id,
            #'--equipment_file', EQUIPMENT_FILE,
            #'--failure_file', FAILURE_FILE,
            #'--maintenance_file', MAINTENANCE_FILE
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        stdout, stderr = process.communicate()
        
        # Clean up temp file
        try:
            os.unlink(temp_csv_path)
        except:
            pass
        
        if process.returncode != 0:
            return jsonify({'error': f'Inference process failed: {stderr}'}), 500
        
        # Extract JSON from stdout (inference.py prints JSON to stdout)
        json_match = re.search(r'--- Inference Results \(JSON\) ---\n(.*?)\n--- Inference Script Finished', 
                              stdout, re.DOTALL)
        
        if not json_match:
            return jsonify({'error': 'Could not parse inference results'}), 500
            
        result = json.loads(json_match.group(1))
        
        # Store prediction only if it's not a simulation
        result_data = result[0]  # First item in the list
        success = False
        result_id = None
        
        if not is_simulation:
            # Only store the prediction if it's not from a simulation
            success, result_id = store_prediction(machine_id, result_data)
            if not success:
                print(f"Warning: Failed to store prediction: {result_id}")
        else:
            print(f"Skipping database storage for simulation prediction")
        
        # Enhanced response with additional details
        response_data = {
            'machine_id': machine_id,
            'status': result_data['status'],
            'failure_probability': result_data['stage1_probability'],
            'timestamp': result_data['timestamp'],
            'failure_predictions': result_data['failure_predictions'],
            'prediction_id': result_id if success and isinstance(result_id, int) else None
        }
        
        # Add days to failure calculation based on recommendations
        if result_data['status'] == 'alert' and result_data['failure_predictions']:
            # Find the highest probability failure
            highest_prob_failure = max(
                result_data['failure_predictions'], 
                key=lambda x: x['detection']['probability']
            )
            
            # Extract action recommendation timeframe if available
            if highest_prob_failure['action_recommendation']['recommended_before']:
                rec_time = datetime.fromisoformat(highest_prob_failure['action_recommendation']['recommended_before'])
                current = datetime.fromisoformat(result_data['timestamp'])
                days_diff = (rec_time - current).days
                response_data['days_to_failure'] = max(0, days_diff)
                response_data['recommended_action'] = highest_prob_failure['action_recommendation']['action']
                response_data['urgency'] = highest_prob_failure['action_recommendation']['urgency']
        
        # Return the enhanced response
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host=os.getenv('FLASK_HOST'), port=int(os.getenv('PREDICTION_AGENT_PORT', 5002)))
