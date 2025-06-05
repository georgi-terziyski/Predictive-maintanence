from flask import Flask, jsonify, request
import psycopg2
import psycopg2.extras  # For dictionary cursor
from dotenv import load_dotenv
import os
import json
from datetime import datetime, timedelta

load_dotenv()

app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def check_db_connection():
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

@app.route('/health')
def health_check():
    db_status = 'healthy' if check_db_connection() else 'unhealthy'
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'database': db_status,
        'version': '1.0'
    })

@app.route('/machines', methods=['GET'])
def get_machines():
    """Get all machines and their status"""
    try:
        conn = get_db_connection()
        # Use DictCursor to get column names in results
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute('SELECT * FROM machines ORDER BY machine_id;')
        machines = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to list of dictionaries
        result = [dict(row) for row in machines]
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['GET'])
def get_predict_data():
    """
    Get prediction data for the last 4 days for a specific machine.
    This endpoint is used by the prediction agent to fetch data for making predictions.
    """
    try:
        # Get machine_id from query parameters (required)
        machine_id = request.args.get('machine_id')
        if not machine_id:
            return jsonify({'error': 'machine_id parameter is required'}), 400
            
        # Calculate date 4 days ago from now
        four_days_ago = datetime.now() - timedelta(days=4)
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query prediction data for the specified machine from the last 4 days
        query = '''
            SELECT * FROM prediction_data
            WHERE machine_id = %s AND timestamp >= %s
            ORDER BY timestamp ASC
        '''
        
        cur.execute(query, (machine_id, four_days_ago))
        prediction_data = cur.fetchall()
        
        # Also get machine information
        cur.execute('SELECT * FROM machines WHERE machine_id = %s', (machine_id,))
        machine_info = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if not machine_info:
            return jsonify({'error': f'No machine found with ID: {machine_id}'}), 404
            
        # Build the response with machine info and prediction data readings
        result = {
            'machine': dict(machine_info),
            'time_period': f'Last 4 days ({four_days_ago.isoformat()} to {datetime.now().isoformat()})',
            'prediction_count': len(prediction_data),
            'prediction_data': [dict(row) for row in prediction_data]
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/live_data', methods=['GET'])
def get_live_data():
    """Get latest sensor data for all machines or last 50 entries for a specific machine"""
    try:
        machine_id = request.args.get('machine_id')
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        if machine_id:
            query = '''
                SELECT * FROM sensor_data
                WHERE machine_id = %s
                ORDER BY timestamp DESC
                LIMIT 50
            '''
            cur.execute(query, (machine_id,))
        else:
            # Get latest reading for each machine
            query = '''
                SELECT s.* FROM sensor_data s
                INNER JOIN (
                    SELECT machine_id, MAX(timestamp) as max_time 
                    FROM sensor_data 
                    GROUP BY machine_id
                ) latest
                ON s.machine_id = latest.machine_id AND s.timestamp = latest.max_time;
            '''
            cur.execute(query)
            
        data = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to list of dictionaries
        result = [dict(row) for row in data]
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/failure_data', methods=['GET'])
def get_failure_data():
    """Get all failure log data for a given machine."""
    try:
        machine_id = request.args.get('machine_id')
        if not machine_id:
            return jsonify({'error': 'machine_id parameter is required'}), 400

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query the failure_logs table, order by timestamp
        query = '''
            SELECT * FROM failure_logs
            WHERE machine_id = %s
            ORDER BY "timestamp" DESC; 
        '''
        
        cur.execute(query, (machine_id,))
        failure_data = cur.fetchall()
        cur.close()
        conn.close()
        
        result = [dict(row) for row in failure_data]
        if not result:
            return jsonify({'message': f'No failure data found for machine_id: {machine_id}', 'data': []}), 200
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/maintenance_history', methods=['GET'])
def get_maintenance_history():
    """Get all maintenance history for a given machine."""
    try:
        machine_id = request.args.get('machine_id')
        if not machine_id:
            return jsonify({'error': 'machine_id parameter is required'}), 400

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query the maintenance_history table, order by timestamp
        query = '''
            SELECT * FROM maintenance_history
            WHERE machine_id = %s
            ORDER BY "timestamp" DESC;
        '''
        
        cur.execute(query, (machine_id,))
        maintenance_data = cur.fetchall()
        cur.close()
        conn.close()
        
        result = [dict(row) for row in maintenance_data]
        if not result:
            return jsonify({'message': f'No maintenance history found for machine_id: {machine_id}', 'data': []}), 200
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/historical_data', methods=['GET'])
def get_historical_data():
    """Get historical sensor data with optional filtering"""
    try:
        # Parse query parameters
        machine_id = request.args.get('machine_id')
        limit = request.args.get('limit', 100, type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        query_params = []
        conditions = []
        
        # Build query based on parameters
        base_query = 'SELECT * FROM sensor_data'
        
        if machine_id:
            conditions.append('machine_id = %s')
            query_params.append(machine_id)
            
        if start_date:
            conditions.append('timestamp >= %s')
            query_params.append(start_date)
            
        if end_date:
            conditions.append('timestamp <= %s')
            query_params.append(end_date)
        
        # Add WHERE clause if conditions exist
        if conditions:
            base_query += ' WHERE ' + ' AND '.join(conditions)
            
        # Add order and limit
        base_query += ' ORDER BY timestamp DESC LIMIT %s'
        query_params.append(limit)
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(base_query, query_params)
        data = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to list of dictionaries
        result = [dict(row) for row in data]
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """Get prediction records with optional filtering"""
    try:
        machine_id = request.args.get('machine_id')
        limit = request.args.get('limit', 100, type=int)
        
        query_params = []
        conditions = []
        
        # Build query based on parameters
        base_query = 'SELECT * FROM predictions'
        
        if machine_id:
            conditions.append('machine_id = %s')
            query_params.append(machine_id)
        
        # Add WHERE clause if conditions exist
        if conditions:
            base_query += ' WHERE ' + ' AND '.join(conditions)
            
        # Add order and limit
        base_query += ' ORDER BY created_at DESC LIMIT %s'
        query_params.append(limit)
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(base_query, query_params)
        data = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to list of dictionaries
        result = [dict(row) for row in data]
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictions', methods=['POST'])
def add_prediction():
    """Store a new prediction from the prediction agent"""
    try:
        data = request.get_json()
        required_fields = ['machine_id', 'status', 'failure_probability', 'prediction_timestamp', 'prediction_details']
        
        # Validate required fields
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Ensure prediction_details is properly formatted for JSONB
        prediction_details = data['prediction_details']
        
        # If prediction_details is not already a string, convert it to a JSON string
        if not isinstance(prediction_details, str):
            try:
                prediction_details = json.dumps(prediction_details)
            except Exception as e:
                return jsonify({'error': f'Failed to convert prediction_details to JSON: {str(e)}'}), 400
        
        print(f"Storing prediction for machine {data['machine_id']} with status {data['status']}")
        
        # Map fields to match database schema
        db_data = {
            'machine_id': data['machine_id'],
            'status': data['status'],
            'confidence': data['failure_probability'],  # Map failure_probability to confidence
            'created_at': data['prediction_timestamp'],  # Map prediction_timestamp to created_at
            'prediction_details': prediction_details
        }
        
        # Prepare SQL statement
        fields = list(db_data.keys())
        placeholders = ['%s'] * len(fields)
        values = [db_data[field] for field in fields]
        
        # Create the INSERT statement
        sql = f'''
            INSERT INTO predictions ({', '.join(fields)})
            VALUES ({', '.join(placeholders)})
            RETURNING id
        '''
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql, values)
        new_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'id': new_id, 'status': 'created'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/maintenance', methods=['GET'])
def get_maintenance():
    """Get maintenance records with optional filtering"""
    try:
        # Parse query parameters
        machine_id = request.args.get('machine_id')
        limit = request.args.get('limit', 100, type=int)
        maintenance_type = request.args.get('type')
        status = request.args.get('status')
        
        query_params = []
        conditions = []
        
        # Build query based on parameters
        base_query = 'SELECT * FROM maintenance'
        
        if machine_id:
            conditions.append('machine_id = %s')
            query_params.append(machine_id)
            
        if maintenance_type:
            conditions.append('maintenance_type = %s')
            query_params.append(maintenance_type)
            
        if status:
            conditions.append('status = %s')
            query_params.append(status)
        
        # Add WHERE clause if conditions exist
        if conditions:
            base_query += ' WHERE ' + ' AND '.join(conditions)
            
        # Add order and limit
        base_query += ' ORDER BY maintenance_date DESC LIMIT %s'
        query_params.append(limit)
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(base_query, query_params)
        data = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to list of dictionaries
        result = [dict(row) for row in data]
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/maintenance', methods=['POST'])
def add_maintenance():
    """Add a new maintenance record"""
    try:
        data = request.get_json()
        required_fields = ['machine_id', 'maintenance_date', 'maintenance_type', 'reason']
        
        # Validate required fields
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare SQL statement with only provided fields
        fields = []
        placeholders = []
        values = []
        
        for key, value in data.items():
            fields.append(key)
            placeholders.append('%s')
            values.append(value)
        
        # Create the INSERT statement
        sql = f'''
            INSERT INTO maintenance ({', '.join(fields)})
            VALUES ({', '.join(placeholders)})
            RETURNING id
        '''
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql, values)
        new_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'id': new_id, 'status': 'created'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/defaults', methods=['GET'])
def get_defaults():
    """Get system defaults with optional category/machine filtering"""
    try:
        category = request.args.get('category')
        machine_id = request.args.get('machine_id')
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        query = 'SELECT * FROM defaults WHERE 1=1'
        params = []
        
        if category:
            query += ' AND category = %s'
            params.append(category)
            
        if machine_id:
            query += ' AND machine_id = %s'
            params.append(machine_id)
        
        cur.execute(query, params)
        data = cur.fetchall()
        cur.close()
        conn.close()
        
        if machine_id:
            # For machine-specific defaults, return a simplified format with averages
            result = {}
            
            # Collect min/max values for averaging
            min_max_values = {}
            
            for row in data:
                row_dict = dict(row)
                key = row_dict['key']
                value = row_dict['value']
                
                # If it's a min/max key, store for later averaging
                if key.endswith('_min') or key.endswith('_max'):
                    base_name = key[:-4]  # Remove _min or _max
                    
                    # Skip vibration, it stays as max only
                    if base_name == 'vibration':
                        if key == 'vibration_max':
                            result[key] = value
                        continue
                    
                    if base_name not in min_max_values:
                        min_max_values[base_name] = {}
                    
                    min_max_values[base_name][key[-3:]] = float(value)
                else:
                    # Non min/max keys go directly into result
                    result[key] = value
            
            # Calculate averages
            for base_name, values in min_max_values.items():
                if 'min' in values and 'max' in values:
                    avg = (values['min'] + values['max']) / 2
                    result[f"{base_name}"] = str(round(avg, 2))
            
            # Include machine info
            machine_info = get_machine_info(machine_id)
            if machine_info:
                result['machine_id'] = machine_id
                result['machine_name'] = machine_info.get('name', '')
                result['machine_type'] = machine_info.get('type', '')
            
            return jsonify(result)
        else:
            # Group by category for general queries (backward compatibility)
            result = {}
            for row in data:
                row_dict = dict(row)
                category = row_dict['category']
                
                if category not in result:
                    result[category] = []
                    
                result[category].append(row_dict)
                
            return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper function to get machine info
def get_machine_info(machine_id):
    """Get basic information about a specific machine"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute('SELECT * FROM machines WHERE machine_id = %s', (machine_id,))
        machine = cur.fetchone()
        cur.close()
        conn.close()
        
        if machine:
            return dict(machine)
        return None
    except:
        return None

@app.route('/simulations', methods=['POST'])
def add_simulation():
    """Store simulation results"""
    try:
        data = request.get_json()
        required_fields = ['machine_id', 'parameters', 'results']
        
        # Validate required fields
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare SQL statement
        fields = ['machine_id', 'parameters', 'results']
        values = [data['machine_id'], json.dumps(data['parameters']), json.dumps(data['results'])]
        
        # Add optional fields if provided
        if 'scenario_type' in data:
            fields.append('scenario_type')
            values.append(data['scenario_type'])
            
        if 'created_by' in data:
            fields.append('created_by')
            values.append(data['created_by'])
        
        # Create the INSERT statement
        placeholders = ['%s'] * len(fields)
        sql = f'''
            INSERT INTO simulations ({', '.join(fields)})
            VALUES ({', '.join(placeholders)})
            RETURNING id
        '''
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql, values)
        new_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'id': new_id, 'status': 'created'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/machine_list', methods=['GET'])
def get_machine_list():
    """Get a simplified list of machine IDs and names only"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute('SELECT machine_id, name FROM machines ORDER BY machine_id;')
        machines = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to list of dictionaries
        result = [dict(row) for row in machines]
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/project_list', methods=['GET'])
def get_project_list():
    """Get a simplified list of project IDs and names only"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute('SELECT id, "project name" FROM projects ORDER BY id;')
        projects = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to list of dictionaries
        result = [dict(row) for row in projects]
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/projects', methods=['POST', 'DELETE'])
def handle_projects():
    """Handle project operations - POST to add, DELETE to remove"""
    if request.method == 'POST':
        return add_project()
    elif request.method == 'DELETE':
        return delete_project()

def add_project():
    """Add a new project"""
    try:
        data = request.get_json()
        
        # Validate required field
        if not data or 'project name' not in data:
            return jsonify({'error': 'Missing required field: project name'}), 400
            
        project_name = data['project name']
        if not project_name or not project_name.strip():
            return jsonify({'error': 'project name cannot be empty'}), 400
        
        # Create the INSERT statement
        sql = '''
            INSERT INTO projects ("project name")
            VALUES (%s)
            RETURNING id
        '''
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql, (project_name.strip(),))
        new_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'id': new_id, 'status': 'created'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def delete_project():
    """Delete a project by name"""
    try:
        # Get project name from query parameters
        project_name = request.args.get('name')
        if not project_name:
            return jsonify({'error': 'Missing required parameter: name'}), 400
            
        if not project_name.strip():
            return jsonify({'error': 'project name cannot be empty'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # First check if project exists
        cur.execute('SELECT id FROM projects WHERE "project name" = %s', (project_name.strip(),))
        existing_project = cur.fetchone()
        
        if not existing_project:
            cur.close()
            conn.close()
            return jsonify({'error': 'Project not found'}), 404
        
        # Delete the project
        cur.execute('DELETE FROM projects WHERE "project name" = %s', (project_name.strip(),))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'status': 'deleted', 'project_name': project_name.strip()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host=os.getenv('FLASK_HOST'), port=int(os.getenv('DATA_AGENT_PORT', 5001)))
