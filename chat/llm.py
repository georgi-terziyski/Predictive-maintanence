import os
import json
from flask import Flask, request, jsonify
import ollama
from datetime import datetime
from instructions import instructions

app = Flask(__name__)

DATA_FOLDER = "data/"  # Folder where JSON files are stored
# MODEL_NAME='llama3:8b'
MODEL_NAME='llama3.2:3b'

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
    user_message = request.json.get("message", "")
    machine_id = request.json.get("machine_id", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Get JSON data using the hardcoded test range
    machine_data = get_latest_json(machine_id)

    bot_response = chat_with_bot(user_message, machine_data)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
