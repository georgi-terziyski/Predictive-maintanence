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
            'fetch_data': '/fetch',
            'machine_list': '/machine_list',
            'project_list': '/project_list',
            'projects': '/projects',
            'defaults': '/defaults',
            'live_data': '/live_data'
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
        # Extract machine_id from request data
        request_data = request.json or {}
        machine_id = request_data.get('machine_id')
        
        if not machine_id:
            return jsonify({'error': 'machine_id is required'}), 400
            
        # Create JSON payload with machine_id
        payload = {'machine_id': machine_id}
        
        # If there's other data in the request, include it in the payload
        for key, value in request_data.items():
            if key != 'machine_id':
                payload[key] = value
        
        # Forward request to prediction agent
        prediction_agent = REGISTERED_AGENTS['prediction_agent']
        response = requests.post(
            f"{prediction_agent['base_url']}{prediction_agent['endpoints']['predict']}",
            json=payload,
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulate', methods=['POST'])
def handle_simulation():
    try:
        request_data = request.json or {}
        
        # Check required parameters - look for machine_id at top level or nested in simulation_data
        machine_id = request_data.get("machine_id", "") or request_data.get("simulation_data", {}).get("machine_id", "")
        if not machine_id:
            return jsonify({'error': 'machine_id is required'}), 400
        
        # If data is nested in simulation_data, flatten it for the simulation agent
        if "simulation_data" in request_data:
            simulation_data = request_data["simulation_data"]
            # Create flattened payload with machine_id at top level
            payload = {
                "machine_id": machine_id,
                "duration_hours": simulation_data.get("duration_hours", 24),
                "initial_values": simulation_data.get("initial_values", {}),
                "fixed_parameters": simulation_data.get("fixed_parameters", {}),
                "interval_minutes": simulation_data.get("interval_minutes", 30)
            }
        else:
            # Data is already flat, use as-is
            payload = request_data
            
        # Forward request to simulation agent
        simulation_agent = REGISTERED_AGENTS['simulation_agent']
        response = requests.post(
            f"{simulation_agent['base_url']}{simulation_agent['endpoints']['simulate']}",
            json=payload,
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/machine_list', methods=['GET'])
def handle_machine_list():
    try:
        # Forward request to data agent
        data_agent = REGISTERED_AGENTS['data_agent']
        response = requests.get(
            f"{data_agent['base_url']}{data_agent['endpoints']['machine_list']}",
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/project_list', methods=['GET'])
def handle_project_list():
    try:
        # Forward request to data agent
        data_agent = REGISTERED_AGENTS['data_agent']
        response = requests.get(
            f"{data_agent['base_url']}{data_agent['endpoints']['project_list']}",
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/projects', methods=['POST'])
def handle_create_project():
    try:
        # Forward request to data agent
        data_agent = REGISTERED_AGENTS['data_agent']
        response = requests.post(
            f"{data_agent['base_url']}{data_agent['endpoints']['projects']}",
            json=request.json,
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/machine_defaults', methods=['GET'])
def handle_machine_defaults():
    try:
        # Get machine_id from request
        machine_id = request.args.get('machine_id')
        if not machine_id:
            return jsonify({'error': 'machine_id parameter is required'}), 400
            
        # Forward request to data agent
        data_agent = REGISTERED_AGENTS['data_agent']
        response = requests.get(
            f"{data_agent['base_url']}{data_agent['endpoints']['defaults']}",
            params={'machine_id': machine_id},
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/live_data', methods=['GET'])
def handle_live_data():
    try:
        # Get machine_id from request if provided
        machine_id = request.args.get('machine_id')
        
        # Forward request to data agent
        data_agent = REGISTERED_AGENTS['data_agent']
        
        # Pass along query parameters if they exist
        params = {}
        if machine_id:
            params['machine_id'] = machine_id
            
        response = requests.get(
            f"{data_agent['base_url']}{data_agent['endpoints']['live_data']}",
            params=params,
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host=os.getenv('FLASK_HOST'),port=int(os.getenv('SUPERVISOR_PORT')))
