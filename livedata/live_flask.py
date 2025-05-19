from flask import Flask, jsonify, request
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import random
import threading
import time
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Constants
SENSOR_FREQ = '30s'  # For documentation
MACHINE_IDS = [f'M{i:03d}' for i in range(1, 11)]

# Get database configuration from environment variables
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

# Ensure timestamps are properly formatted
def ensure_format(df, col='Timestamp'):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
    return df

# Create necessary database tables
def create_tables():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create sensor_data table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            Timestamp TIMESTAMP,
            Machine_ID TEXT,
            AFR FLOAT,
            Current FLOAT,
            Pressure FLOAT, 
            RPM INT,
            Temperature FLOAT,
            Vibration FLOAT
        );
    """)
    
    # Create failure_logs table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS failure_logs (
            Machine_ID TEXT,
            Timestamp TIMESTAMP,
            Failure_Type TEXT
        );
    """)
    
    # Create maintenance_history table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS maintenance_history (
            Machine_ID TEXT,
            Timestamp TIMESTAMP,
            Maintenance_Action TEXT
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

# Save data to PostgreSQL database
def save_to_postgres(table_name, df):
    if df.empty:
        return
        
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Ensure timestamps are properly formatted
    df = ensure_format(df)
    
    # Build the INSERT query
    columns = ', '.join(df.columns)
    placeholders = ', '.join(['%s'] * len(df.columns))
    
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    
    # Execute for each row
    for _, row in df.iterrows():
        cur.execute(query, tuple(row))
    
    conn.commit()
    cur.close()
    conn.close()

# Generate synthetic sensor data
def generate_sensor_data(failure_df, maintenance_df):
    sensor_data = []
    
    for machine in MACHINE_IDS:
        now = pd.Timestamp.now()
        
        # Ensure timestamps in failure_df and maintenance_df are properly formatted
        if not failure_df.empty and 'Timestamp' in failure_df.columns:
            failure_times = failure_df[failure_df['Machine_ID'] == machine]['Timestamp'].values
            failure_times = pd.to_datetime(failure_times)
        else:
            failure_times = []
            
        if not maintenance_df.empty and 'Timestamp' in maintenance_df.columns:
            maintenance_times = maintenance_df[maintenance_df['Machine_ID'] == machine]['Timestamp'].values
            maintenance_times = pd.to_datetime(maintenance_times)
        else:
            maintenance_times = []
        
        # Create dictionary mapping failure times to maintenance times
        downtime_intervals = {}
        if len(failure_times) == len(maintenance_times):
            downtime_intervals = dict(zip(failure_times, maintenance_times))
        
        # Skip machines that are in downtime
        if any(start <= now < end for start, end in downtime_intervals.items()):
            continue
        
        # Generate sensor readings with some correlation between parameters
        afr = round(np.random.uniform(10, 15), 2)
        current = round(np.random.uniform(20, 35), 2)
        pressure = round(np.random.uniform(3, 6), 2)
        rpm = random.randint(2800, 3600)
        temperature = round(np.random.uniform(65, 100), 2)
        vibration = round(np.random.uniform(1, 9), 2)
        
        # Add correlations between parameters
        if vibration > 6:
            current += np.random.uniform(2, 5)
        if temperature > 85:
            pressure += np.random.uniform(0.5, 1.5)
        if afr < 11:
            temperature += np.random.uniform(3, 6)
        
        # Add to sensor data list
        sensor_data.append([now, machine, afr, current, pressure, rpm, temperature, vibration])
    
    # Create DataFrame from collected data
    return pd.DataFrame(sensor_data, 
                       columns=['Timestamp', 'Machine_ID', 'AFR', 'Current', 
                                'Pressure', 'RPM', 'Temperature', 'Vibration'])

# Continuous sensor data generation loop (to be run in a background thread)
def sensor_data_loop():
    # Initialize empty DataFrames for failure and maintenance records
    failure_df = pd.DataFrame(columns=['Machine_ID', 'Timestamp', 'Failure_Type'])
    maintenance_df = pd.DataFrame(columns=['Machine_ID', 'Timestamp', 'Maintenance_Action'])
    
    # Counter for batch simulation
    batch_counter = 0
    
    while True:
        try:
            # Ensure timestamps are properly formatted
            failure_df = ensure_format(failure_df)
            maintenance_df = ensure_format(maintenance_df)
            
            # Generate new sensor data
            df = generate_sensor_data(failure_df, maintenance_df)
            
            # Save to database
            save_to_postgres('sensor_data', df)
            
            # Simulate 2-week interval in real time (30s interval => 40320 batches per 2 weeks)
            batch_counter += 1
            failures = []
            maintenance = []
            
            # Every 2 weeks, potentially generate failures and maintenance events
            if batch_counter % 40320 == 0:  # Every 2 weeks (40320 * 30s = 2 weeks)
                for machine in MACHINE_IDS:
                    machine_df = df[df['Machine_ID'] == machine]
                    
                    # Find anomalous readings
                    anomaly = machine_df[
                        (machine_df['Temperature'] > 85) |
                        (machine_df['Vibration'] > 6) |
                        (machine_df['Pressure'] > 5)
                    ]
                    
                    # If anomalies exist, potentially create a failure record
                    if not anomaly.empty:
                        sampled = anomaly.sample(1).iloc[0]
                        failure_type = random.choice([
                            'Bearing Failure', 'Motor Burnout', 'Overheating',
                            'Pressure System Failure', 'Carbon Buildup', 'Electrical Malfunction'
                        ])
                        
                        # Record failure
                        failures.append([sampled['Machine_ID'], sampled['Timestamp'], failure_type])
                        
                        # Map failure types to appropriate maintenance actions
                        maintenance_action_map = {
                            'Bearing Failure': 'Bearing Replacement',
                            'Motor Burnout': 'Motor Servicing',
                            'Overheating': 'Cooling System Check',
                            'Pressure System Failure': 'Pressure Adjustment',
                            'Carbon Buildup': 'Carbon Cleanup',
                            'Electrical Malfunction': 'Electrical Inspection'
                        }
                        
                        # Record maintenance (30 minutes after failure)
                        maintenance.append([
                            sampled['Machine_ID'],
                            sampled['Timestamp'] + timedelta(minutes=30),
                            maintenance_action_map[failure_type]
                        ])
            
            # Save any new failures to database
            if failures:
                failure_df = pd.DataFrame(failures, columns=['Machine_ID', 'Timestamp', 'Failure_Type'])
                failure_df = ensure_format(failure_df)
                save_to_postgres('failure_logs', failure_df)
            
            # Save any new maintenance records to database
            if maintenance:
                maintenance_df = pd.DataFrame(maintenance, columns=['Machine_ID', 'Timestamp', 'Maintenance_Action'])
                maintenance_df = ensure_format(maintenance_df)
                save_to_postgres('maintenance_history', maintenance_df)
            
            # Log the data generation (every 10th generation to avoid spamming)
            if batch_counter % 10 == 0:
                print(f"[{datetime.now()}] Generated sensor data for {len(df)} machines. Batch #{batch_counter}")
                
            # Sleep for 30 seconds before next batch
            time.sleep(30)
        except Exception as e:
            print(f"Error in sensor data loop: {str(e)}")
            time.sleep(30)  # Still wait before retrying

# API Routes

@app.route('/sensor-data/<machine_id>')
def get_sensor_data_by_machine(machine_id):
    """Get sensor data for a specific machine"""
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

@app.route('/failures/<machine_id>')
def get_failures_by_machine(machine_id):
    """Get failure records for a specific machine"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        query = """
            SELECT * FROM failure_logs
            WHERE Machine_ID = %s
            ORDER BY Timestamp DESC
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

@app.route('/maintenance/<machine_id>')
def get_maintenance_logs(machine_id):
    """Get maintenance records for a specific machine"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        query = """
            SELECT * FROM maintenance_history
            WHERE Machine_ID = %s
            ORDER BY Timestamp DESC
            LIMIT 10
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
        "machines": MACHINE_IDS
    })

# This is a workaround since @app.before_first_request is deprecated in Flask 2.0+
# We'll start the background thread directly before running the app

# Initialize database tables when the script runs
if __name__ == '__main__':
    # Create tables before starting the app
    create_tables()
    
    # Start the background thread for data generation
    data_thread = threading.Thread(target=sensor_data_loop)
    data_thread.daemon = True  # Thread will exit when main thread exits
    data_thread.start()
    print(f"Started background sensor data generation thread")
    
    # Get port from environment or use default
    port = int(os.getenv('LIVEDATA_PORT', 5006))
    
    print(f"Starting live data generator on port {port}")
    print(f"Generating data for machines: {', '.join(MACHINE_IDS)}")
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=port, debug=False)
