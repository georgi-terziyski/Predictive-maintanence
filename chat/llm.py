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

def get_latest_json(machine_id=None):
    """Finds the latest JSON file within the hardcoded test range. If commented out, defaults to the latest file."""
    json_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".json")]
    machine_files = {}

    for file in json_files:
        parts = file.replace(".json", "").split("_")  # Extract machine_id and date
        if len(parts) == 2:
            m_id, date_str = parts
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if m_id not in machine_files or file_date > machine_files[m_id][1]:
                    machine_files[m_id] = (file, file_date)
            except ValueError:
                continue  # Skip invalid files  

    if machine_id:
        # Return latest JSON for the specific machine_id within the date range
        if machine_id in machine_files:
            file_path = os.path.join(DATA_FOLDER, machine_files[machine_id][0])
            with open(file_path, "r") as f:
                return json.load(f)
    else:
        # Return the latest JSON file overall within the date range
        latest_file = max(machine_files.values(), key=lambda x: x[1], default=None)
        if latest_file:
            file_path = os.path.join(DATA_FOLDER, latest_file[0])
            with open(file_path, "r") as f:
                return json.load(f)

    return None  # No valid JSON file found in the given date range

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
    if any(word in user_message for word in ["simulate failure", "simulate conditions",
        "test machine behavior", "run simulation","start simulation", "create synthetic conditions", 
        "simulate scenarios", "run test conditions"]):
        print('in simulation')
        simulation_data = request.json.get("simulation data")
        payload = {"duration": simulation_data} if simulation_data else {}
        response = requests.post(f"{SUPERVISOR_API_URL}/simulate", json=payload)
        return jsonify(response.json()), response.status_code
    # Default chatbot response
    machine_data = get_latest_json(machine_id)
    bot_response = chat_with_bot(user_message, machine_data)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(host=os.getenv('FLASK_HOST'), port=5005, debug=True)
