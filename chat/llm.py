import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import requests
import ollama
import fitz  # PyMuPDF
from dotenv import load_dotenv
# Embedding & ChromaDB
import chromadb
from sentence_transformers import SentenceTransformer

# Instructions
from instructions import instructions

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)

# Environment vars
MODEL_NAME = os.getenv('MODEL_NAME')
SUPERVISOR_API_URL = os.getenv('SUPERVISOR_API_URL')
DATA_FOLDER = os.getenv('DATA_FOLDER', './data')
os.makedirs(DATA_FOLDER, exist_ok=True)

# ChromaDB & Embeddings setup
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("knowledge_base")
embedder = SentenceTransformer("all-MiniLM-L6-v2",token=False)

# Allowed file types
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def retrieve_relevant_context(query: str, top_k: int = 3):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    return "\n---\n".join(docs)

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

def chat_with_bot(user_input):
    # machine_context = json.dumps(machine_data, indent=2) if machine_data else "No machine data available."
    rag_context = retrieve_relevant_context(user_input)

    prompt = f"""Use the following machine data and knowledge base context to answer the user question.

Knowledge Base:
{rag_context}

User Question:
{user_input}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt}
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
    if "__system_health" in user_message:
        try:
            response = requests.get(f"{SUPERVISOR_API_URL}/health")
            return jsonify(response.json()), response.status_code
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Prediction request
    if "__predict_failure" in user_message:
        response = requests.post(
            f"{SUPERVISOR_API_URL}/predict",
            headers={"Content-Type": "application/json"},
            json={"machine_id":machine_id}
        )
        return jsonify(response.json()), response.status_code
    
    # Simulation request
    if "__simulation_run" in user_message:
        simulation_data = request.json.get("simulation_data")
        machine_id = simulation_data.get("machine_id")
        if not machine_id:
            return jsonify({"error": "machine_id is required for simulation -- from llm"}), 400
        payload = {"simulation_data": simulation_data} if simulation_data else {}
        response = requests.post(f"{SUPERVISOR_API_URL}/simulate", json=payload)
        return jsonify(response.json()), response.status_code
    

    bot_response = chat_with_bot(user_message)
    return jsonify({"response": bot_response})


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(DATA_FOLDER, filename)
        file.save(file_path)

        # Extract text from PDF
        text = extract_text_from_pdf(file_path)

        # Embed and store the whole PDF as one document
        embedding = embedder.encode(text).tolist()
        doc_id = filename  # or f"{filename}_full"
        collection.add(documents=[text], embeddings=[embedding], ids=[doc_id])

        return jsonify({"message": f"{filename} uploaded and indexed successfully."})
    
    return jsonify({"error": "Unsupported file type"}), 400

if __name__ == "__main__":
    app.run(host=os.getenv('FLASK_HOST', '0.0.0.0'), port=int(os.getenv('CHAT_AGENT_PORT', 5005)), debug=True, use_reloader=False)
