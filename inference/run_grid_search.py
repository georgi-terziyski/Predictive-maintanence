# run_grid_search.py

import subprocess # To run other scripts
import os
import pandas as pd
from datetime import datetime

# --- Configuration for Grid Search ---
PYTHON_EXECUTABLE = "python"                  # Or the specific path to python interpreter (e.g., "D:/Anaconda3/envs/tensorrt/python.exe")
SCRIPT_TO_RUN     = "predictive_model.py" 
RESULTS_FILE      = 'grid_search_results.csv' # Must match the one in predictive_model.py

# Define the ranges for W (Window hours) and H (Horizon hours)
# Example ranges:
W_VALUES = [ 6,  9, 12, 24, 36, 48, 60]       # e.g., 12h, 24h, 36h, 48h windows
H_VALUES = [12, 24, 36, 48, 60, 72, 84, 96]   # e.g., predict 12h, 24h, 36h, 48h ahead

# --- Grid Search Execution ---
if __name__ == "__main__":
    print("--- Starting Grid Search for Optimal W and H ---")
    
    # Optional: Backup or clear previous results file
    if os.path.exists(RESULTS_FILE):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{RESULTS_FILE}.bak_{timestamp}"
        try:
            os.rename(RESULTS_FILE, backup_file)
            print(f"Backed up previous results to: {backup_file}")
        except OSError as e:
            print(f"Could not back up results file: {e}. Appending results.")
            
    # Check if the script to run exists
    if not os.path.exists(SCRIPT_TO_RUN):
        print(f"Error: Script '{SCRIPT_TO_RUN}' not found in the current directory.")
        exit()

    total_runs = len(W_VALUES) * len(H_VALUES)
    current_run = 0

    # Iterate through all combinations
    for w in W_VALUES:
        for h in H_VALUES:
            current_run += 1
            print(f"\n--- Running ({current_run}/{total_runs}): W={w}h, H={h}h ---")
            
            # Construct the command to run the predictive model script
            command = [
                PYTHON_EXECUTABLE,
                SCRIPT_TO_RUN,
                "--window", str(w),
                "--horizon", str(h)
                # NO --plot argument here, so plots will be disabled by default
            ]
            
            # Execute the command
            try:
                # Use subprocess.run for better control and error handling
                process = subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"--- Finished W={w}h, H={h}h ---")
                # Print stdout and stderr for debugging if needed
                # print("STDOUT:\n", process.stdout)
                # print("STDERR:\n", process.stderr)
                
            except subprocess.CalledProcessError as e:
                print(f"!!! ERROR running W={w}h, H={h}h !!!")
                print(f"Command failed with return code: {e.returncode}")
                print("STDERR:")
                print(e.stderr)
                print("STDOUT:")
                print(e.stdout)
                print("--- Skipping to next combination ---")
                continue # Continue to the next iteration even if one fails
            except Exception as e:
                 print(f"!!! An unexpected error occurred running W={w}h, H={h}h: {e} !!!")
                 print("--- Skipping to next combination ---")
                 continue

    print("\n--- Grid Search Complete ---")

    # --- Post-processing: Analyze Results ---
    if os.path.exists(RESULTS_FILE):
        print(f"\n--- Analyzing Results from {RESULTS_FILE} ---")
        try:
            results_df = pd.read_csv(RESULTS_FILE)
            print("Grid Search Results Summary:")
            # Sort by Max_F1 score descending
            results_df_sorted = results_df.sort_values(by='Max_F1', ascending=False)
            print(results_df_sorted.to_string()) # Print the full sorted table

            # Find the best run based on Max_F1
            if not results_df_sorted.empty:
                best_run = results_df_sorted.iloc[0]
                print("\n--- Best Run Found (based on Max F1-Score) ---")
                print(f"W = {best_run['W_Hours']}h, H = {best_run['H_Hours']}h")
                print(f"Max F1 = {best_run['Max_F1']:.4f} (at Threshold ≈ {best_run['Optimal_Threshold']:.3f})")
                print(f"  Precision @ Max F1 = {best_run['Precision_at_Max_F1']:.4f}")
                print(f"  Recall @ Max F1 = {best_run['Recall_at_Max_F1']:.4f}")
                print(f"  ROC AUC = {best_run['ROC_AUC']:.4f}")
                print(f"  PR AUC  = {best_run['PR_AUC' ]:.4f}")
                print(f"  Bayesian F1 = {best_run['Bayes_F1']:.4f} (P={best_run['Bayes_Precision']:.4f}, R={best_run['Bayes_Recall']:.4f})")
            else:
                print("Results file is empty or could not be parsed.")

        except Exception as e:
            print(f"Error reading or analyzing results file: {e}")
    else:
        print(f"Results file '{RESULTS_FILE}' not found.")
