import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import openai
from chromadb import PersistentClient

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)

# Configs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "openaikey"
MODEL_NAME = os.getenv('OPENAI_MODEL') or "gpt-3.5-turbo"
SUPERVISOR_API_URL = os.getenv('SUPERVISOR_API_URL')
CHROMA_DB_PATH = "chroma_store"

# OpenAI and ChromaDB setup
openai.api_key = OPENAI_API_KEY
chroma_client = PersistentClient(path=CHROMA_DB_PATH)
summary_collection = chroma_client.get_or_create_collection(name="pdf_summaries")

# Fetch live machine data
def get_live_data(machine_id):
    try:
        response = requests.get(f"{SUPERVISOR_API_URL}/live-data/{machine_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return None

# RAG: Retrieve relevant summaries
def get_relevant_summaries(query, top_k=3):
    embedding_response = openai.Embedding.create(
        input=[query],
        model="text-embedding-ada-002"
    )
    query_embedding = embedding_response["data"][0]["embedding"]

    search_results = summary_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents"]
    )
    documents = search_results["documents"][0]
    return "\n\n---\n\n".join(documents)

# RAG + machine context response
def generate_answer_with_context(context, machine_data, question):
    machine_info = json.dumps(machine_data, indent=2) if machine_data else "No live machine data available."
    prompt = f"""You are a helpful assistant answering the question based strictly on the provided information.

Context:
{context}

Machine Data:
{machine_info}

Question: {question}
"""
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response["choices"][0]["message"]["content"]

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip().lower()
    machine_id = request.json.get("machine_id", "").strip()

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Health check
    if "__system_health" in user_message:
        try:
            response = requests.get(f"{SUPERVISOR_API_URL}/health")
            return jsonify(response.json()), response.status_code
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Prediction
    if "__predict_failure" in user_message:
        try:
            response = requests.post(
                f"{SUPERVISOR_API_URL}/predict",
                headers={"Content-Type": "application/json"},
                json={}
            )
            return jsonify(response.json()), response.status_code
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Simulation
    if "__simulation_run" in user_message:
        simulation_data = request.json.get("simulation_data")
        payload = {"simulation_data": simulation_data} if simulation_data else {}
        try:
            response = requests.post(f"{SUPERVISOR_API_URL}/simulate", json=payload)
            return jsonify(response.json()), response.status_code
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Default: RAG-based chatbot with optional machine context
    try:
        machine_data = get_live_data(machine_id) if machine_id else None
        context = get_relevant_summaries(user_message)
        answer = generate_answer_with_context(context, machine_data, user_message)
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host=os.getenv('FLASK_HOST', '0.0.0.0'), port=5005, debug=True)
