from flask import Flask, jsonify
import psycopg2
from dotenv import load_dotenv
import os
from datetime import datetime

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
    except:
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

@app.route('/live_data', methods=['GET'])
def get_live_data():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM machine_data ORDER BY timestamp DESC LIMIT 1;')
        data = cur.fetchone()
        cur.close()
        conn.close()
        return jsonify({
            'machine_id': data[0],
            'timestamp': data[1],
            'temperature': data[2],
            'vibration': data[3],
            'load': data[4]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/historical_data', methods=['GET'])
def get_historical_data():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM machine_data ORDER BY timestamp DESC LIMIT 100;')
        data = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify([{
            'machine_id': row[0],
            'timestamp': row[1],
            'temperature': row[2],
            'vibration': row[3],
            'load': row[4]
        } for row in data])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=int(os.getenv('DATA_AGENT_PORT')))
