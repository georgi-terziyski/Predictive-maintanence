import os
from dotenv import load_dotenv
import json
from flask import Flask, request, jsonify
import ollama
import requests
from datetime import datetime
from instructions import instructions

load_dotenv()

app = Flask(__name__)

MODEL_NAME = os.getenv('MODEL_NAME')
SUPERVISOR_API_URL = os.getenv('SUPERVISOR_API_URL')
DATA_FOLDER = os.getenv('DATA_FOLDER')

def get_live_data(machine_id):
    """Fetches live data from the supervisor API for a specific machine_id."""
    try:
        response = requests.get(f"{SUPERVISOR_API_URL}/live-data/{machine_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return None


def chat_with_bot(user_input, machine_data):
    """Generates a response using the chatbot and machine data."""
    machine_context = json.dumps(machine_data, indent=2) if machine_data else "No machine data available."
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"{machine_context}\n\nUser Question: {user_input}"}
        ]
    )
    return response['message']['content']

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").lower()
    machine_id = request.json.get("machine_id", "")
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Health check
    if any(word in user_message for word in ["system health", "check health", "supervisor status", "check status", "is system running"]):
        print('in health')
        try:
            response = requests.get(f"{SUPERVISOR_API_URL}/health")
            return jsonify(response.json()), response.status_code
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Prediction request
    if any(word in user_message for word in ["predict failure", "failure risk", "breakdown prediction", "estimate failure", "next failure", "time to failure"]):
        print('in predict')
        response = requests.post(
            f"{SUPERVISOR_API_URL}/predict",
            headers={"Content-Type": "application/json"},
            json={}  # Sending an empty JSON payload
            )
        return jsonify(response.json()), response.status_code
    
    # Simulation request (with optional duration)
    if any(word in user_message for word in ["__simulation_run"]):
        print('in simulation')
        simulation_data = request.json.get("simulation_data")
        payload = {"duration": simulation_data} if simulation_data else {}
        response = requests.post(f"{SUPERVISOR_API_URL}/simulate", json=payload)
        return jsonify(response.json()), response.status_code
    
    # Default chatbot response
    if machine_id:
        machine_data = get_live_data(machine_id)
        if not machine_data:
            return jsonify({"error": f"No live data available for machine ID: {machine_id}"}), 404
    else:
        machine_data = None

    bot_response = chat_with_bot(user_message, machine_data)
    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(host=os.getenv('FLASK_HOST'), port=5005, debug=True)
