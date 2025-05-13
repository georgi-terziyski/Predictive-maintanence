# run_stage2_grid_search_WH.py

import subprocess 
import os
import pandas as pd
from datetime import datetime

# --- Configuration ---
PYTHON_EXECUTABLE = "python" # Adjust if needed (e.g., "path/to/your/python.exe")
SCRIPT_TO_RUN = "stage2_classify_type.py" # The script we just finalized
RESULTS_FILE = 'stage2_grid_search_WH_results.csv' # Output file name (NO FFT)

# --- Define Grid Search Parameters ---
# Focus around the best W found earlier (48-72h) and explore H2
#W2_VALUES = [6, 9, 12, 24, 36, 48, 60, 72, 84] 
#H2_VALUES = [6, 9, 12, 24, 36, 48, 60, 72] 
W2_VALUES = [6, 9, 12, 24, 36, 48, 60, 72] 
H2_VALUES = [84] 

# --- Grid Search Execution ---

if __name__ == "__main__":
    print("--- Starting Stage 2 Grid Search for Optimal W2 and H2 (No FFT) ---")
    
    # Backup previous results
    if os.path.exists(RESULTS_FILE):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{RESULTS_FILE}.bak_{timestamp}"
        try: os.rename(RESULTS_FILE, backup_file); print(f"Backed up previous results to: {backup_file}")
        except OSError as e: print(f"Could not back up results file: {e}. Appending.")
            
    if not os.path.exists(SCRIPT_TO_RUN): print(f"Error: Script '{SCRIPT_TO_RUN}' not found."); exit()

    total_runs = len(W2_VALUES) * len(H2_VALUES)
    current_run = 0

    # Iterate through all combinations
    for w2 in W2_VALUES:
        for h2 in H2_VALUES:
            current_run += 1
            print(f"\n--- Running ({current_run}/{total_runs}): Stage 2 with W2={w2}h, H2={h2}h ---")
            
            # Construct the command 
            # NO --fft flag, NO --plot flag
            command = [ PYTHON_EXECUTABLE, SCRIPT_TO_RUN, "--window", str(w2), "--horizon", str(h2) ]
            
            try: # Execute the command
                process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8') 
                print(f"--- Finished W2={w2}h, H2={h2}h ---")
                # Optional: Print short summary from stdout
                # print(process.stdout.split('--- Evaluating Stage 2 Model ---')[-1][:500]) # Show evaluation part
                
            except subprocess.CalledProcessError as e:
                print(f"!!! ERROR running W2={w2}h, H2={h2}h !!!"); print(f"Return code: {e.returncode}")
                print("STDERR:"); print(e.stderr); print("STDOUT:"); print(e.stdout)
                print("--- Skipping to next combination ---"); continue 
            except Exception as e:
                print(f"!!! Unexpected error running W2={w2}h, H2={h2}h: {e} !!!")
                print("--- Skipping to next combination ---"); continue

    print("\n--- Stage 2 Grid Search Complete ---")

    # --- Analyze Results ---
    if os.path.exists(RESULTS_FILE):
        print(f"\n--- Analyzing Results from {RESULTS_FILE} ---")
        try:
            results_df_s2 = pd.read_csv(RESULTS_FILE)
            print("Stage 2 Grid Search Results Summary:")
            # Sort by Macro F1 score descending
            results_df_sorted_s2 = results_df_s2.sort_values(by='F1_Macro_S2', ascending=False)
            # Display relevant columns
            display_cols = ['W_Hours_S2', 'H_Hours_S2', 'Accuracy_S2', 'F1_Macro_S2', 'Recall_Macro_S2', 'Precision_Macro_S2']
            print(results_df_sorted_s2[display_cols].to_string(index=False)) 

            if not results_df_sorted_s2.empty:
                best_run_s2 = results_df_sorted_s2.iloc[0]
                print("\n--- Best Run Found for Stage 2 (based on Macro F1-Score) ---")
                print(f"W2 = {int(best_run_s2['W_Hours_S2'])}h, H2 = {int(best_run_s2['H_Hours_S2'])}h")
                print(f"Macro F1 = {best_run_s2['F1_Macro_S2']:.4f}")
                print(f"  Macro Precision = {best_run_s2['Precision_Macro_S2']:.4f}")
                print(f"  Macro Recall = {best_run_s2['Recall_Macro_S2']:.4f}")
                print(f"  Overall Accuracy = {best_run_s2['Accuracy_S2']:.4f}")
            else: print("Results file empty.")
        except Exception as e: print(f"Error reading/analyzing results file: {e}")
    else: print(f"Results file '{RESULTS_FILE}' not found.")
