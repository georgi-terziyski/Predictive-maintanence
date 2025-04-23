from flask import Flask, request, jsonify, Response
import os
import subprocess
import glob

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
    
    with open(progress_file, 'w') as f:
        f.truncate(0)
    if os.name == 'nt':
        python_exec = os.path.join('venv', 'Scripts', 'python.exe')
    else:
        python_exec = os.path.join('venv', 'bin', 'python')

    process = subprocess.Popen(
    [python_exec, 'predictive_model.py', '--save_model'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

    output_lines = []
    with open(progress_file, 'a') as progress:
        for line in process.stdout:
            cleaned = line.strip()
            if cleaned:
                progress.write(f"{cleaned}\n")
                output_lines.append(cleaned)

        for line in process.stderr:
            cleaned = line.strip()
            if cleaned:
                progress.write(f"ERROR: {cleaned}\n")
                output_lines.append(f"ERROR: {cleaned}")

    process.wait()

    if process.returncode == 0:
        cleanup_uploads()
    
    last_line = output_lines[-1] if output_lines else "No output."
    return jsonify({'last_output': last_line}), 200


progress_stages = {
    "Loading Data": "--- Loading Data ---",
    "Creating Features": "--- Creating Features",
    "Creating Target": "--- Creating Target Variable",
    "Training Model": "--- Training Model",
    "Evaluating Model": "--- Evaluating Model ---",
    "Applying Bayesian Filter": "--- Applying Bayesian Filter",
    "Saving Predictions": "--- Saving Stage 1 Predictions",
    "Run Complete": "--- predictive_model.py Script Finished ---"
}

@app.route('/progress', methods=['GET'])
def get_progress():
    progress_file = 'projects/progress.txt'

    if not os.path.exists(progress_file):
        return jsonify({'error': 'Progress file not found'}), 404

    with open(progress_file, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    progress_percent = (total_lines / 77) * 100  # 77 line in progress.txt
    current_step = "Starting..."

    for step, marker in progress_stages.items():
        if any(marker in line for line in lines):
            current_step = step

    return jsonify({
        'progress': min(progress_percent, 100),
        'status': current_step,
        'lines_read': total_lines
    }), 200
if __name__ == '__main__':
    app.run(debug=True,port=5010, threaded=True)