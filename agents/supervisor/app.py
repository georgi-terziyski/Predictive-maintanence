from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
from datetime import datetime
import requests

load_dotenv()

app = Flask(__name__)

# Registered agents (would normally be in database)
REGISTERED_AGENTS = {
    'data_agent': {
        'base_url': os.getenv('DATA_AGENT_URL'),
        'endpoints': {
            'health': '/health',
            'fetch_data': '/fetch'
        }
    },
    'prediction_agent': {
        'base_url': os.getenv('PREDICTION_AGENT_URL'),
        'endpoints': {
            'health': '/health',
            'predict': '/predict'
        }
    },
    'simulation_agent': {
        'base_url': os.getenv('SIMULATION_AGENT_URL'),
        'endpoints': {
            'health': '/health',
            'simulate': '/simulate'
        }
    },
    'analytics_agent': {
        'base_url': os.getenv('ANALYTICS_AGENT_URL'),
        'endpoints': {
            'health': '/health',
            'analyze': '/analyze',
            'compare': '/compare'
        }
    }
}

@app.route('/health')
def health_check():
    # Check health of all registered agents
    agent_status = {}
    for agent_name, agent_config in REGISTERED_AGENTS.items():
        try:
            response = requests.get(
                f"{agent_config['base_url']}{agent_config['endpoints']['health']}",
                timeout=2
            )
            agent_status[agent_name] = response.json().get('status', 'unknown')
        except:
            agent_status[agent_name] = 'unreachable'
            
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0',
        'agents': agent_status
    })

@app.route('/predict', methods=['POST'])
def handle_prediction():
    try:
        # Forward request to prediction agent
        prediction_agent = REGISTERED_AGENTS['prediction_agent']
        response = requests.post(
            f"{prediction_agent['base_url']}{prediction_agent['endpoints']['predict']}",
            json=request.json,
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulate', methods=['POST'])
def handle_simulation():
    try:
        # Forward request to simulation agent
        simulation_agent = REGISTERED_AGENTS['simulation_agent']
        response = requests.post(
            f"{simulation_agent['base_url']}{simulation_agent['endpoints']['simulate']}",
            json=request.json,
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def handle_analysis():
    try:
        # Forward request to analytics agent
        analytics_agent = REGISTERED_AGENTS['analytics_agent']
        response = requests.post(
            f"{analytics_agent['base_url']}{analytics_agent['endpoints']['analyze']}",
            json=request.json,
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def handle_comparison():
    try:
        # Forward request to analytics agent
        analytics_agent = REGISTERED_AGENTS['analytics_agent']
        response = requests.post(
            f"{analytics_agent['base_url']}{analytics_agent['endpoints']['compare']}",
            json=request.json,
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=int(os.getenv('SUPERVISOR_PORT')))
