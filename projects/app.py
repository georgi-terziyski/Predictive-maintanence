from flask import Flask, request, jsonify, Response
import os
import subprocess
import glob
import shutil
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'projects/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def cleanup_uploads():
    upload_path = os.path.join('.', 'projects', 'uploads', '*')
    for file in glob.glob(upload_path):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

@app.route('/upload', methods=['POST'])
def upload_files():
    if not request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    saved_files = []

    for file in request.files.values():
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            saved_files.append(file.filename)

    if not saved_files:
        return jsonify({'error': 'No valid CSV files uploaded'}), 400

    return jsonify({'success': f'{len(saved_files)} CSV file(s) uploaded', 'files': saved_files}), 200


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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

FINAL_MODELS_DIR = os.path.join(PROJECT_ROOT, 'Predictive-maintanence', 'projects', 'final_models')
INFERENCE_DIR = os.path.join(PROJECT_ROOT, 'Predictive-maintanence', 'inference')
HISTORY_DIR = os.path.join(INFERENCE_DIR, 'history')

FILES_TO_PROCESS = ['feature_columns_xgb_s1.joblib', 'predictive_model_xgb_s1.joblib', 'stage2_class_encoder.joblib', 'stage2_features_W84_H84_temp.joblib', 'stage2_model_W84_H84_temp.joblib']

@app.route('/deploy-model', methods=['POST'])
def update_models():
    try:
        # Create timestamped folder under history
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        archive_dir = os.path.join(HISTORY_DIR, timestamp)
        os.makedirs(archive_dir, exist_ok=True)

        # Move old models to history
        for filename in FILES_TO_PROCESS:
            current_model_path = os.path.join(INFERENCE_DIR, filename)
            archive_path = os.path.join(archive_dir, filename)

            if os.path.exists(current_model_path):
                shutil.move(current_model_path, archive_path)

        # Copy new models from final_models to inference
        for filename in FILES_TO_PROCESS:
            new_model_path = os.path.join(FINAL_MODELS_DIR, filename)
            target_path = os.path.join(INFERENCE_DIR, filename)

            if os.path.exists(new_model_path):
                shutil.copy(new_model_path, target_path)
                os.remove(new_model_path)
            else:
                return jsonify({"error": f"New model not found: {new_model_path}"}), 404

        return jsonify({
            "message": "Models updated successfully",
            "archived_to": archive_dir
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,port=5010)