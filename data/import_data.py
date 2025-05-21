import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv
import time
import json

# Load environment variables
load_dotenv()

# Database connection parameters
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    return psycopg2.connect(**DB_CONFIG)

def check_db_connection():
    """Test the database connection"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT 1')
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return False

def import_machines_from_csv(csv_file):
    """
    Extract unique machine IDs from the CSV file and insert them into the machines table
    Returns a list of imported machine IDs
    """
    print(f"Reading machine data from {csv_file}...")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract unique machine IDs
    unique_machines = df['Machine_ID'].unique()
    
    # Connect to the database
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Insert each unique machine
    for machine_id in unique_machines:
        cur.execute(
            """
            INSERT INTO machines 
            (machine_id, name, type, location, status, installation_date) 
            VALUES (%s, %s, %s, %s, %s, %s) 
            ON CONFLICT (machine_id) DO NOTHING
            """,
            (
                machine_id, 
                f"Machine {machine_id}", 
                "Industrial Pump", 
                f"Building {machine_id[-1]}", 
                "active",
                "2023-01-01"
            )
        )
    
    # Commit the transaction
    conn.commit()
    print(f"✅ Imported {len(unique_machines)} machines: {', '.join(unique_machines)}")
    
    # Close the connection
    cur.close()
    conn.close()
    
    return list(unique_machines)

def import_sensor_data_from_csv(csv_file, batch_size=1000):
    """
    Import sensor data from CSV to the sensor_data table
    batch_size: Number of records to insert in a single transaction
    test_mode: If True, only imports the first batch
    """
    start_time = time.time()
    print(f"Importing sensor data from {csv_file}...")
    
    # Read the CSV file
    df = pd.read_csv(csv_file) 
    total_rows = len(df)
    print(f"Found {total_rows} sensor readings in the CSV file")
    
    # Connect to the database
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Process data in batches
    batches = range(0, len(df), batch_size)
    for i, batch_start in enumerate(batches):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        # Prepare data for insertion
        data = []
        for _, row in batch_df.iterrows():
            data.append((
                row['Machine_ID'],
                row['Timestamp'],
                row['AFR'],
                row['Current'],
                row['Pressure'],
                row['RPM'],
                row['Temperature'],
                row['Vibration']
            ))
        
        # Use execute_values for efficient batch insertion
        execute_values(
            cur,
            """
            INSERT INTO sensor_data 
            (machine_id, timestamp, afr, current, pressure, rpm, temperature, vibration)
            VALUES %s
            ON CONFLICT (machine_id, timestamp) DO NOTHING
            """,
            data
        )
        
        # Commit batch
        conn.commit()
        
        # Print progress
        progress = (batch_end / total_rows) * 100
        elapsed = time.time() - start_time
        print(f"Progress: {progress:.1f}% - Processed {batch_end}/{total_rows} records in {elapsed:.1f} seconds")
    
    # Close the connection
    cur.close()
    conn.close()
    
    # Print summary
    total_time = time.time() - start_time
    print(f"✅ Imported sensor data in {total_time:.1f} seconds ({total_rows / total_time:.1f} records/sec)")

def generate_sample_predictions(machine_ids):
    """Generate sample prediction records for the specified machines"""
    print("Generating sample prediction data...")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    count = 0
    for machine_id in machine_ids:
        # Create two sample predictions per machine (one recent, one older)
        predictions = [
            # Most recent prediction (3 days from now)
            {
                'machine_id': machine_id,
                'predicted_failure_date': pd.Timestamp.now() + pd.Timedelta(days=30),
                'confidence': 0.78,
                'model_version': '1.0.0',
                'prediction_details': json.dumps({
                    'failing_component': 'bearing',
                    'contributing_factors': ['high vibration', 'temperature fluctuation'],
                    'recommendation': 'Schedule maintenance within 3 weeks'
                })
            },
            # Older prediction (from a week ago, longer failure window)
            {
                'machine_id': machine_id,
                'created_at': pd.Timestamp.now() - pd.Timedelta(days=7),
                'predicted_failure_date': pd.Timestamp.now() + pd.Timedelta(days=45),
                'confidence': 0.65,
                'model_version': '1.0.0',
                'prediction_details': json.dumps({
                    'failing_component': 'bearing',
                    'contributing_factors': ['high vibration'],
                    'recommendation': 'Monitor closely'
                })
            }
        ]
        
        # Insert each prediction
        for pred in predictions:
            fields = list(pred.keys())
            placeholders = ', '.join(['%s'] * len(fields))
            values = [pred[field] for field in fields]
            
            query = f"""
                INSERT INTO predictions ({', '.join(fields)})
                VALUES ({placeholders})
                ON CONFLICT DO NOTHING
            """
            
            cur.execute(query, values)
            count += 1
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"✅ Generated {count} sample prediction records")

def generate_sample_maintenance(machine_ids):
    """Generate sample maintenance records for the specified machines"""
    print("Generating sample maintenance data...")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    count = 0
    for machine_id in machine_ids:
        # Create a sample maintenance record for each machine
        maintenance = {
            'machine_id': machine_id,
            'maintenance_date': pd.Timestamp.now() - pd.Timedelta(days=45),
            'completion_date': pd.Timestamp.now() - pd.Timedelta(days=45),
            'maintenance_type': 'preventive',
            'reason': 'Scheduled quarterly maintenance',
            'work_performed': 'Bearing lubrication, belt inspection, calibration',
            'technician_name': 'John Smith',
            'technician_comments': 'All systems operating within normal parameters',
            'parts_replaced': None,
            'status': 'completed',
            'downtime_hours': 2.5,
            'cost': 350.00
        }
        
        fields = list(maintenance.keys())
        placeholders = ', '.join(['%s'] * len(fields))
        values = [maintenance[field] for field in fields]
        
        query = f"""
            INSERT INTO maintenance ({', '.join(fields)})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
        """
        
        cur.execute(query, values)
        count += 1
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"✅ Generated {count} sample maintenance records")

def main():
    """Main function to import data into the database"""
    print("Starting data import process...")
    
    # Test database connection
    if not check_db_connection():
        print("❌ Cannot connect to the database. Please check your connection settings.")
        return
    
    print("✅ Successfully connected to the database")
    
    # Determine CSV file path
    csv_file = "sensor_data.csv"
    
    # Import machines
    machine_ids = import_machines_from_csv(csv_file)
    
    # Import sensor data
    batch_size = 5000
    import_sensor_data_from_csv(csv_file, batch_size)
    
    # Generate sample predictions and maintenance data
    generate_sample = input("Generate sample prediction and maintenance data? (y/n): ").lower() == 'y'
    if generate_sample:
        generate_sample_predictions(machine_ids)
        generate_sample_maintenance(machine_ids)
    
    print("Data import process completed successfully!")

if __name__ == "__main__":
    main()
