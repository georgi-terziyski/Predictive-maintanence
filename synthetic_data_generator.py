import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import psycopg2
import psycopg2.extras
from psycopg2 import extras
from dotenv import load_dotenv
import threading
import time
import requests
from flask import Flask, jsonify, request
from scipy.stats import gamma as gamma_dist
import logging
import os.path

# Load environment variables
load_dotenv()

# --- Logging Setup ---
# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/synthetic_data.log'),
        logging.StreamHandler()  # This will also print to console
    ]
)
logger = logging.getLogger(__name__)

# --- Flask Application Setup ---
app = Flask(__name__)

# --- Database Configuration ---
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

# --- Configuration Constants ---
MACHINE_IDS = [f'M{i:03d}' for i in range(1, 11)]  # 10 machines
DATA_INTERVAL_SECONDS = 30  # Generate data every 30 seconds
PREDICTION_INTERVAL_ITERATIONS = 60  # Save to prediction_data every 60 iterations (30 minutes)

# --- Trend Configuration for Anomalies ---
TREND_CONFIG = {
    'Bearing Failure': {
        'sensors': {'Vibration': 2.0, 'Current': 2.5, 'Temperature': 4.0},
        'trend_type': 'quadratic', 
        'start_ratio': 0.1,
        'probability': 0.02
    },
    'Motor Burnout': {
        'sensors': {'Current': 6.5, 'Temperature': 9.0, 'Vibration': 0.7},
        'trend_type': 'linear', 
        'start_ratio': 0.4,
        'probability': 0.02
    },
    'Overheating': {
        'sensors': {'Temperature': 10.0, 'Pressure': 1.0, 'Current': 1.5},
        'trend_type': 'linear',
        'start_ratio': 0.2,
        'probability': 0.03
    },
    'Pressure System Failure': {
        'sensors': {'Pressure': 1.5, 'Vibration': 0.5},
        'trend_type': 'linear',
        'start_ratio': 0.2,
        'probability': 0.02
    },
    'Carbon Buildup': {
        'sensors': {'AFR': -1.5, 'Temperature': 5.0},
        'trend_type': 'linear',
        'start_ratio': 0.05,
        'probability': 0.03
    },
    'Electrical Malfunction': {
        'sensors': {'Current': 5.0, 'Vibration': 1.5},
        'trend_type': 'spikes', 
        'start_ratio': 0.6,
        'spike_probability': 0.18,
        'probability': 0.03
    }
}

# --- State Management ---
# Track active anomalies
active_anomalies = {machine: None for machine in MACHINE_IDS}
# Counter for tracking when to write to prediction_data
data_counter = 0

# --- Database Functions ---
def get_db_connection():
    """Create a database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        return conn
    except psycopg2.Error as e:
        logger.error(f"Error connecting to PostgreSQL database: {e}")
        raise

def clamp_integer(value):
    """Ensure integer values are within PostgreSQL INTEGER type limits"""
    if pd.isna(value):
        return None  # This shouldn't happen with our constraints, but just in case
    max_int = 2147483647
    min_int = -2147483647
    return max(min(int(value), max_int), min_int)

def save_to_sensor_data(sensor_df):
    """Insert sensor data into the sensor_data table"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Prepare data for insertion
        sensor_data = sensor_df.copy()
        
        # Build tuples for insertion
        tuples = [
            (row['Timestamp'], row['Machine_ID'], 
             row['AFR'], row['Current'], row['Pressure'], 
             clamp_integer(row['RPM']), row['Temperature'], row['Vibration'])
            for _, row in sensor_data.iterrows()
        ]
        
        # Use executemany with psycopg2.extras for better performance
        extras.execute_batch(cursor, """
            INSERT INTO sensor_data 
            (Timestamp, Machine_ID, AFR, Current, Pressure, RPM, Temperature, Vibration)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (Machine_ID, Timestamp) DO UPDATE SET
            AFR = EXCLUDED.AFR,
            Current = EXCLUDED.Current,
            Pressure = EXCLUDED.Pressure,
            RPM = EXCLUDED.RPM,
            Temperature = EXCLUDED.Temperature,
            Vibration = EXCLUDED.Vibration
        """, tuples)
        
        conn.commit()
        logger.info(f"Saved {len(tuples)} records to sensor_data table")
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error when saving to sensor_data: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

def trigger_prediction(machine_id):
    """Trigger a prediction for a specific machine via the prediction agent"""
    try:
        prediction_agent_url = os.getenv('PREDICTION_AGENT_URL', 'http://localhost:5002')
        response = requests.post(
            f"{prediction_agent_url}/predict", 
            json={"machine_id": machine_id},
            timeout=10  # Longer timeout as predictions can take time
        )
        
        if response.status_code == 200:
            prediction_result = response.json()
            # Log details about the prediction
            status = prediction_result.get('status')
            probability = prediction_result.get('failure_probability')
            logger.info(f"Prediction for {machine_id}: status={status}, probability={probability:.4f}")
            return True
        else:
            logger.warning(f"Failed to trigger prediction for {machine_id}: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error triggering prediction for {machine_id}: {str(e)}")
        return False

def save_to_prediction_data(sensor_df):
    """Insert sensor data into the prediction_data table and trigger predictions"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Prepare data for insertion
        sensor_data = sensor_df.copy()
        
        # Build tuples for insertion
        tuples = [
            (row['Machine_ID'], row['Timestamp'], 
             row['AFR'], row['Current'], row['Pressure'], 
             clamp_integer(row['RPM']), row['Temperature'], row['Vibration'])
            for _, row in sensor_data.iterrows()
        ]
        
        # Use executemany with psycopg2.extras for better performance
        extras.execute_batch(cursor, """
            INSERT INTO prediction_data 
            (machine_id, timestamp, afr, current, pressure, rpm, temperature, vibration)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (machine_id, timestamp) DO UPDATE SET
            afr = EXCLUDED.afr,
            current = EXCLUDED.current,
            pressure = EXCLUDED.pressure,
            rpm = EXCLUDED.rpm,
            temperature = EXCLUDED.temperature,
            vibration = EXCLUDED.vibration
        """, tuples)
        
        conn.commit()
        logger.info(f"Saved {len(tuples)} records to prediction_data table")
        
        # After successful save, trigger predictions for each machine
        logger.info("Triggering predictions for all machines with new data")
        successful_predictions = 0
        failed_predictions = 0
        
        # Get unique machine IDs from the dataframe
        unique_machines = sensor_data['Machine_ID'].unique()
        
        for machine_id in unique_machines:
            # Add a small delay between predictions to avoid overwhelming the prediction agent
            if successful_predictions + failed_predictions > 0:
                time.sleep(2)  # 2 second delay between machines
                
            # Use threading to not block the data generation
            thread = threading.Thread(
                target=lambda m=machine_id: trigger_prediction(m),
                daemon=True
            )
            thread.start()
            
            # We're not waiting for thread completion here to keep the data generation flowing
            logger.info(f"Started prediction thread for machine {machine_id}")
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error when saving to prediction_data: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# --- Data Generation Functions ---
def generate_base_sensor_data():
    """Generate base sensor data with normal operating ranges"""
    now = datetime.now()
    sensor_data = []
    
    for machine in MACHINE_IDS:
        # Normal operating ranges (relatively stable)
        afr = round(np.random.uniform(11.5, 14.5), 2) 
        current = round(np.random.uniform(22, 32), 2) 
        pressure = round(np.random.uniform(3.5, 5.0), 2) 
        rpm_base = random.randint(3000, 3400) 
        temperature = round(np.random.uniform(68, 82), 2) 
        vibration = round(np.random.uniform(1.5, 4.5), 2) 
        
        # Add correlations between parameters
        if vibration > 3.5:
            current += round(np.random.uniform(0.5, 1.5), 2)
        if temperature > 78:
            pressure += round(np.random.uniform(0.1, 0.3), 2)
        if afr < 12.5:
            temperature += round(np.random.uniform(1.0, 2.0), 2)
        
        # Base RPM variance 
        rpm = rpm_base + random.randint(-50, 50)
        
        sensor_data.append({
            'Timestamp': now,
            'Machine_ID': machine,
            'AFR': afr,
            'Current': current,
            'Pressure': pressure,
            'RPM': rpm,
            'Temperature': temperature,
            'Vibration': vibration
        })
    
    return pd.DataFrame(sensor_data)

def start_new_anomaly(machine_id):
    """Start a new anomaly trend for a specific machine"""
    failure_types = list(TREND_CONFIG.keys())
    selected_type = random.choice(failure_types)
    
    # Calculate how long this anomaly will last - between 30 minutes and 4 hours
    duration_minutes = random.randint(30, 240)
    # Avoid division by zero, ensuring at least 1 iteration
    minutes_per_iteration = DATA_INTERVAL_SECONDS / 60
    iterations = max(1, int(duration_minutes / minutes_per_iteration))
    
    return {
        'type': selected_type,
        'remaining_iterations': iterations,
        'progress': 0.0,
        'config': TREND_CONFIG[selected_type]
    }

def apply_anomaly_trends(sensor_df):
    """Apply ongoing anomaly trends to the sensor data"""
    global active_anomalies
    modified_df = sensor_df.copy()
    
    # For each machine, check if we should start a new anomaly or continue an existing one
    for machine in MACHINE_IDS:
        # Decide if we should start a new anomaly
        if active_anomalies[machine] is None:
            # Random chance to start a new anomaly
            if random.random() < 0.005:  # 0.5% chance per data point
                active_anomalies[machine] = start_new_anomaly(machine)
                logger.info(f"Started new anomaly '{active_anomalies[machine]['type']}' for {machine}")
        
        # If there's an active anomaly for this machine, apply it
        if active_anomalies[machine] is not None:
            anomaly = active_anomalies[machine]
            config = anomaly['config']
            
            # Get the row for this machine
            machine_mask = modified_df['Machine_ID'] == machine
            if not machine_mask.any():
                continue
            
            # Calculate progress for this iteration (0 to 1)
            total_iterations = max(1, anomaly['remaining_iterations'] + anomaly['progress'] * anomaly['remaining_iterations'])
            progress = anomaly['progress'] + (1 / total_iterations)
            progress = min(progress, 1.0)  # Cap at 1.0
            
            # Only start applying changes after the start_ratio point
            if progress >= config['start_ratio']:
                effective_progress = (progress - config['start_ratio']) / (1 - config['start_ratio'])
                effective_progress = min(max(effective_progress, 0.0), 1.0)
                
                # Apply changes to each affected sensor
                for sensor, max_delta in config['sensors'].items():
                    # Skip if sensor not in DataFrame
                    if sensor not in modified_df.columns:
                        continue
                    
                    # Calculate delta based on trend type
                    delta = 0
                    if config['trend_type'] == 'linear':
                        delta = max_delta * effective_progress
                    elif config['trend_type'] == 'quadratic':
                        delta = max_delta * (effective_progress ** 2)
                    elif config['trend_type'] == 'exponential':
                        # Cap exponential growth to avoid extreme values
                        exp_factor = min(effective_progress * 4, 4)
                        delta = max_delta * (np.exp(exp_factor) - 1) / (np.exp(4) - 1)
                    elif config['trend_type'] == 'spikes':
                        if random.random() < config.get('spike_probability', 0.1):
                            delta = max_delta * random.uniform(0.5, 1.0)
                    
                    # Apply delta to the sensor value
                    if delta != 0:
                        current_value = modified_df.loc[machine_mask, sensor].values[0]
                        noise = np.random.normal(0, abs(max_delta * 0.05))
                        new_value = current_value + delta + noise
                        
                        # Round appropriately
                        if sensor in ['AFR', 'Current', 'Pressure', 'Temperature', 'Vibration']:
                            modified_df.loc[machine_mask, sensor] = round(new_value, 2)
                        else:
                            modified_df.loc[machine_mask, sensor] = new_value
            
            # Update the anomaly progress or end it if completed
            anomaly['progress'] = progress
            anomaly['remaining_iterations'] -= 1
            
            if anomaly['remaining_iterations'] <= 0:
                logger.info(f"Completed anomaly '{anomaly['type']}' for {machine}")
                active_anomalies[machine] = None
    
    return modified_df

# --- Main Data Generation Loop ---
def data_generation_loop():
    """Background thread function to generate data at regular intervals"""
    global data_counter
    
    logger.info("Starting data generation loop")
    
    while True:
        try:
            # Generate base sensor data
            sensor_df = generate_base_sensor_data()
            
            # Apply any active anomaly trends
            sensor_df = apply_anomaly_trends(sensor_df)
            
            # Save to sensor_data table
            save_to_sensor_data(sensor_df)
            
            # Increment counter and check if we need to save to prediction_data
            data_counter += 1
            if data_counter >= PREDICTION_INTERVAL_ITERATIONS:
                logger.info("Reached 30-minute mark, saving to prediction_data")
                save_to_prediction_data(sensor_df)
                data_counter = 0
            
            # Sleep for the configured interval
            time.sleep(DATA_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"Error in data generation loop: {e}")
            time.sleep(DATA_INTERVAL_SECONDS)  # Still wait before retrying

# --- Flask Routes ---
@app.route('/health')
def health_check():
    """Health check endpoint"""
    db_status = "connected"
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "running",
        "database": db_status,
        "timestamp": datetime.now().isoformat(),
        "data_generation_counter": data_counter,
        "active_anomalies": {k: v['type'] if v else None for k, v in active_anomalies.items()},
        "next_prediction_data_in_minutes": (PREDICTION_INTERVAL_ITERATIONS - data_counter) * (DATA_INTERVAL_SECONDS / 60)
    })

@app.route('/sensor-data/<machine_id>')
def get_sensor_data_by_machine(machine_id):
    """Get recent sensor data for a specific machine"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        query = """
            SELECT * FROM sensor_data 
            WHERE Machine_ID = %s 
            ORDER BY Timestamp DESC 
            LIMIT 50
        """
        
        cur.execute(query, (machine_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to list of dictionaries
        result = [dict(row) for row in rows]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/prediction-data/<machine_id>')
def get_prediction_data_by_machine(machine_id):
    """Get prediction data for a specific machine"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        query = """
            SELECT * FROM prediction_data 
            WHERE machine_id = %s 
            ORDER BY timestamp DESC 
            LIMIT 20
        """
        
        cur.execute(query, (machine_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to list of dictionaries
        result = [dict(row) for row in rows]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get statistics about the data generation"""
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "data_counter": data_counter,
        "machines": MACHINE_IDS,
        "active_anomalies": {k: v['type'] if v else None for k, v in active_anomalies.items()},
        "minutes_until_next_prediction": (PREDICTION_INTERVAL_ITERATIONS - data_counter) * (DATA_INTERVAL_SECONDS / 60)
    })

# --- Main Execution ---
if __name__ == '__main__':
    # Create a background thread for data generation
    data_thread = threading.Thread(target=data_generation_loop)
    data_thread.daemon = True  # Thread will exit when main thread exits
    data_thread.start()
    
    # Start the Flask application
    port = int(os.getenv('SYNTHETIC_DATA_PORT', 5006))
    logger.info(f"Starting synthetic data generator on port {port}")
    logger.info(f"Generating data for machines: {', '.join(MACHINE_IDS)}")
    app.run(host='0.0.0.0', port=port, debug=False)
