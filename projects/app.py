from flask import Flask, request, jsonify, Response
import os
import ollama
import pandas as pd
import subprocess
import glob
import shutil
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

PROJECT_NAME= os.getenv('PROJECT_NAME')
FINAL_MODELS_DIR = os.path.join(PROJECT_ROOT, PROJECT_NAME, 'projects', 'final_models')
INFERENCE_DIR = os.path.join(PROJECT_ROOT, PROJECT_NAME, 'inference')
ALLOWED_EXTENSIONS = {'csv', 'pdf'}
HISTORY_DIR = os.path.join(INFERENCE_DIR, 'history')
MODEL_NAME = os.getenv('MODEL_NAME')
UPLOAD_FOLDER = 'projects/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DATACENTER_COLUMNS = {"timestamp", "cpu_usage", "cpu_temp", "ram_usage", "disk_io", "smart_health",
                      "air_temp", "humidity", "fan_rpm", "ups_load", "pue", "status", "severity", "failure_type", "message"}
TRAINING_COLUMNS = {"machine_id", "timestamp", "maintenance_action", "afr", "current", "pressure",
                    "rpm", "temperature", "vibration", "equipment_age (years)", "usage_cycles", "failure_type"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_columns_from_file(file_path):
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, nrows=1)
            return list(df.columns)
        elif file_path.endswith(".pdf"):
            import fitz
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            # crude heuristic: words starting with capital letter as columns
            return [w.strip() for w in text.split() if w.isalpha() and w[0].isupper()]
        else:
            return []
    except:
        return []

def classify_file_by_columns(columns):
    cols_lower = set(c.lower() for c in columns)
    datacenter_overlap = len(cols_lower & DATACENTER_COLUMNS)
    training_overlap = len(cols_lower & TRAINING_COLUMNS)
    if datacenter_overlap >= 2:
        return "DATACENTER"
    elif training_overlap >= 2:
        return "TRAINING"
    else:
        return "LLM_FALLBACK"

def cleanup_uploads():
    upload_path = os.path.join('.', 'projects', 'uploads', '*')
    for file in glob.glob(upload_path):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     if not request.files: return jsonify({'error': 'No files uploaded'}), 400

#     saved_files = []
#     for file in request.files.values():
#         if file and file.filename.endswith('.csv'):
#             filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#             file.save(filepath)
#             saved_files.append(file.filename)

#             df = pd.read_csv(filepath, nrows=1)
#             columns = set(df.columns.str.lower())
#             datacenter_overlap = len(columns & set(c.lower() for c in DATACENTER_COLUMNS))
#             training_overlap = len(columns & set(c.lower() for c in TRAINING_COLUMNS))

#             if datacenter_overlap >= 2:
#                 classification = "DATACENTER"
#             elif training_overlap >= 2:
#                 classification = "TRAINING"
#             else:
#                 prompt = f"Decide if this file is TRAINING or DATACENTER. Columns: {list(df.columns)}. Answer with one word."
#                 resp = ollama.chat(model=MODEL_NAME, messages=[{"role":"user","content":prompt}])
#                 classification = resp['message']['content'].strip().upper()

#             # Move file to classified folder
#             target_folder = os.path.join('projects', classification.lower())
#             os.makedirs(target_folder, exist_ok=True)
#             shutil.move(filepath, os.path.join(target_folder, file.filename))

#     if not saved_files: return jsonify({'error': 'No valid CSV files uploaded'}), 400
#     return jsonify({'success': f'{len(saved_files)} file(s) uploaded and classified.', 'files': saved_files}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    columns = extract_columns_from_file(file_path)
    decision = classify_file_by_columns(columns)

    # LLM fallback if columns don't match
    if decision == "LLM_FALLBACK":
        print('in LLM')
        prompt = f"""
        You are a file classifier. Decide if this file is for:
        1. TRAINING (machine maintenance, usage, failure logs),
        2. DATACENTER (system telemetry like cpu_usage, ram_usage, pue, etc.).

        File columns: {columns}

        Answer with exactly one word: TRAINING or DATACENTER.
        """
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        decision = response['message']['content'].strip().upper()

    # --- Branch based on decision ---
    if decision == "TRAINING":
        # do training-specific processing
        # e.g., save for predictive model pipeline
        return jsonify({
            "filename": filename,
            "columns": columns,
            "classification": decision,
            "message": "File saved for TRAINING pipeline."
        })

    elif decision == "DATACENTER":
        # do datacenter-specific processing
        # e.g., store for telemetry ingestion
        return jsonify({
            "filename": filename,
            "columns": columns,
            "classification": decision,
            "message": "File saved for DATACENTER ingestion."
        })

    else:
        return jsonify({"error": "Could not classify file"}), 400

@app.route('/run-script', methods=['POST'])
def run_script():
    progress_file = 'projects/progress.txt'

    # Clear previous progress
    with open(progress_file, 'w') as f:
        f.truncate(0)

    if os.name == 'nt':
        python_exec = os.path.join('venv', 'Scripts', 'python.exe')
    else:
        python_exec = os.path.join('venv', 'bin', 'python')

    output_lines = []

    def run_and_log(cmd):
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        with open(progress_file, 'a') as progress:
            for line in process.stdout:
                cleaned = line.strip()
                if cleaned:
                    progress.write(f"{cleaned}\n")
                    progress.flush()
                    output_lines.append(cleaned)

            for line in process.stderr:
                cleaned = line.strip()
                if cleaned:
                    progress.write(f"ERROR: {cleaned}\n")
                    progress.flush()
                    output_lines.append(f"ERROR: {cleaned}")

        process.wait()
        return process.returncode

    # Run predictive_model.py
    returncode1 = run_and_log([python_exec, '-u', 'predictive_model.py', '--save_model'])

    if returncode1 == 0:
        # Only run second stage if the first one succeeds
        returncode2 = run_and_log([python_exec, '-u', 'stage2_classify_type.py'])
        if returncode2 == 0:
            cleanup_uploads()
            move_models()
    else:
        output_lines.append("Stage 1 failed. Skipping Stage 2.")

    last_line = output_lines[-1] if output_lines else "No output."
    return jsonify({'last_output': last_line}), 200


@app.route('/progress', methods=['GET'])
def get_progress():
    progress_file = 'projects/progress.txt'

    CHECKPOINTS = [
        ("Loading Data", 5),
        ("Creating Features", 15),
        ("Training Model", 30),
        ("Evaluating Model", 40),
        ("Stage 1 Finished", 50),
        ("Running Stage 2", 55),
        ("Creating Features", 70),
        ("Training Stage 2", 85),
        ("Stage 2 Finished", 100),
    ]

    try:
        with open(progress_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        current_step = "Starting..."
        percent = 0

        for label, pct in reversed(CHECKPOINTS):
            if any(label in line for line in lines):
                current_step = label
                percent = pct
                break

        return jsonify({
            'step': current_step,
            'progress': percent
        }), 200

    except FileNotFoundError:
        return jsonify({'step': 'No progress yet.', 'progress': 0}), 200
    

# @app.route('/move_models', methods=['POST'])
def move_models():
    # Set destination folder (make sure it exists)
    target_dir = 'projects/final_models'
    os.makedirs(target_dir, exist_ok=True)

    files_to_move = [
        'projects/script_result/stage1/predictive_model_xgb_s1.joblib',
        'projects/script_result/stage1/feature_columns_xgb_s1.joblib',
        'projects/script_result/stage2/stage2_model_W84_H84_temp.joblib',
        'projects/script_result/stage2/stage2_features_W84_H84_temp.joblib',
        'projects/script_result/stage2/stage2_class_encoder.joblib'
    ]

    moved_files = []

    for file_path in files_to_move:
        if os.path.exists(file_path):
            dest_path = os.path.join(target_dir, os.path.basename(file_path))
            shutil.move(file_path, dest_path)
            moved_files.append(os.path.basename(file_path))
        else:
            moved_files.append(f"{os.path.basename(file_path)} (not found)")
            
     # Clean up any leftover files in stage1 and stage2
    for folder in ['projects/script_result/stage1', 'projects/script_result/stage2']:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    return jsonify({
        'message': 'Files moved to archive.',
        'files': moved_files
    }), 200


FILES_TO_PROCESS = [
    'feature_columns_xgb_s1.joblib',
    'predictive_model_xgb_s1.joblib',
    'stage2_class_encoder.joblib',
    'stage2_features_W84_H84_temp.joblib',
    'stage2_model_W84_H84_temp.joblib'
]

@app.route('/deploy-model', methods=['POST'])
def update_models():
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        archive_dir = os.path.join(HISTORY_DIR, timestamp)
        os.makedirs(archive_dir, exist_ok=True)

        moved_files = []
        skipped_archive = []
        copied_files = []
        newly_added = []

        # Move existing inference models to history
        for filename in FILES_TO_PROCESS:
            current_model_path = os.path.join(INFERENCE_DIR, filename)
            archive_path = os.path.join(archive_dir, filename)

            if os.path.exists(current_model_path):
                shutil.move(current_model_path, archive_path)
                moved_files.append(filename)
            else:
                skipped_archive.append(filename)

        # Copy from final_models to inference
        for filename in FILES_TO_PROCESS:
            new_model_path = os.path.join(FINAL_MODELS_DIR, filename)
            target_path = os.path.join(INFERENCE_DIR, filename)

            if os.path.exists(new_model_path):
                shutil.copy(new_model_path, target_path)
                os.remove(new_model_path)
                copied_files.append(filename)
                if filename in skipped_archive:
                    newly_added.append(filename)
            else:
                return jsonify({"error": f"New model not found: {new_model_path}"}), 404

        return jsonify({
            "message": "Models deployed successfully",
            "archived_to": archive_dir,
            "archived_files": moved_files,
            "newly_added_files": newly_added,
            "copied_to_inference": copied_files
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,port=5010)