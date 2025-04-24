# stage2_classify_type.py (v. Cleaned - NO FFT, Fix Duplicates, Fix Save)

import pandas as pd
import numpy as np
import joblib 
import os
from   datetime import datetime, timedelta
import warnings
from   sklearn.model_selection import train_test_split 
from   sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from   sklearn.preprocessing import LabelEncoder
from   sklearn.utils.class_weight import compute_sample_weight 
# --- NO FFT IMPORTS ---
# from numpy.fft import fft 
# import cupy as cp # No CuPy needed now
import xgboost as xgb 
import matplotlib.pyplot as plt
import argparse 
import csv 

# --- Configure Warnings ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) 
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning) 

# Import necessary functions from predictive_model library
try:
    # create_features should be the version WITHOUT FFT but WITH slope, skew, kurt, std_diff, corr
    from predictive_model import ( load_data, create_features, apply_bayesian_filter, 
                                   DATA_DIR, SENSOR_FILE, FAILURE_FILE, EQUIPMENT_FILE, MAINTENANCE_FILE )
    print("Successfully imported functions from predictive_model.py")
except ImportError as e: print(f"Import Error: {e}")           ; exit()
except Exception   as e: print(f"Unexpected Import Error: {e}"); exit()
    
# --- Configuration for Stage 2 ---
PREDICTION_HORIZON_HOURS_S1_REF     = 48 
DEFAULT_FEATURE_WINDOW_HOURS_S2     = 84 # Default W2 based on last best result
DEFAULT_PREDICTION_HORIZON_HOURS_S2 = 84 # Default H2

STAGE1_PREDICTIONS_FILE = 'projects/script_result/stage1/stage1_predictions.csv'

# Paths for Stage 2 artifacts
# Note: fft suffix removed from template names
STAGE2_MODEL_SAVE_PATH_TEMPLATE    = 'projects/script_result/stage2/stage2_model_W{w}_H{h}_temp.joblib' 
STAGE2_FEATURES_SAVE_PATH_TEMPLATE = 'projects/script_result/stage2/stage2_features_W{w}_H{h}_temp.joblib'
STAGE2_CLASSES_SAVE_PATH           = 'projects/script_result/stage2/stage2_class_encoder.joblib' 

# Output file for Stage 2 Grid Search Results
STAGE2_RESULTS_FILE = 'projects/script_result/stage2/stage2_grid_search_WH_results.csv' # Use non-FFT results file now

STAGE2_TEST_SIZE = 0.3
BAYES_ALPHA = 0.7; BAYES_THRESHOLD = 0.75; BAYES_STEPS = 2
# --- NO FFT_BINS ---

# --- NO FFT FUNCTIONS (calculate_fft_features, _calculate_fft_bins, add_fft_features_manual) ---

# --- FUNCTION to Create Stage 2 Target (Using H2) ---
def create_target_stage2(df_features, failure_df, prediction_horizon_hrs_s2, 
                          stage1_risky_df, target_col_name='Actual_Failure_Type'):
    """ Creates the multi-class target for Stage 2 data. """
    # ... (Full function code - NO CHANGES NEEDED from last correct version) ...
    print(f"\n--- Creating Stage 2 Target '{target_col_name}' (H2={prediction_horizon_hrs_s2}h) ---")
    if not isinstance(df_features.index, pd.DatetimeIndex): print("Error: df_features index must be DatetimeIndex."); return None, None
    if stage1_risky_df.empty: print("Error: stage1_risky_df is empty."); return None, None
    if failure_df is None: failure_df = pd.DataFrame() 
    # Use unique indices directly from the potentially cleaned df_features
    common_indices = df_features.index # Assume df_features passed here is already filtered and cleaned
    if len(common_indices) == 0: print("Error: No common indices passed."); return None, None
    df_s2 = df_features.copy(); print(f"Processing {len(df_s2)} instances for Stage 2 target.")
    try: # Add H1 Actual target safely
         risky_preds_reindexed = stage1_risky_df['Failure_Within_H'].reindex(common_indices)
         df_s2['Failure_Within_H1_Actual'] = risky_preds_reindexed.fillna(0).astype(int)
    except Exception as e: print(f"Warning: Could not reindex H1 target: {e}. Setting to 0."); df_s2['Failure_Within_H1_Actual'] = 0
    df_s2[target_col_name] = 'Normal' 
    le = None 
    if not failure_df.empty: 
        print(f"Mapping actual failure types (H2={prediction_horizon_hrs_s2}h)..."); horizon_delta_s2 = timedelta(hours=prediction_horizon_hrs_s2) 
        failure_df_sorted = failure_df.sort_values('Timestamp').set_index('Timestamp'); 
        indices_to_map = df_s2.index # Use the index of the df passed to the function
        for idx in indices_to_map: # Mapping logic using H2
             try: machine_id = df_s2.loc[idx, 'Machine_ID']; 
             except KeyError: print(f"Warning: Machine_ID missing at {idx}."); continue
             if isinstance(machine_id, pd.Series): machine_id = machine_id.iloc[0] 
             window_end_s2 = idx + horizon_delta_s2 
             machine_failures = failure_df_sorted[ (failure_df_sorted['Machine_ID'] == machine_id) & (failure_df_sorted.index > idx) & (failure_df_sorted.index <= window_end_s2) ] 
             if not machine_failures.empty: first_failure_type = machine_failures['Failure_Type'].iloc[0]; df_s2.loc[idx, target_col_name] = first_failure_type
        print(f"Type distribution (H2={prediction_horizon_hrs_s2}h):\n{df_s2[target_col_name].value_counts()}")
    else: print("Warning: Failure log empty.")
    try: # Encode labels
        le = LabelEncoder() ; df_s2['Target_Encoded'] = le.fit_transform(df_s2[target_col_name])
        print("\nEncoded classes:"); print(dict(zip(le.classes_, le.transform(le.classes_))))
        try: joblib.dump(le, STAGE2_CLASSES_SAVE_PATH); print(f"LE saved.")
        except Exception as e_save: print(f"Error saving LE: {e_save}")
    except Exception as e_enc: print(f"Error encoding labels: {e_enc}"); return None, None 
    return df_s2, le 

# --- FUNCTION for Stage 2 Training (Corrected Save Logic) ---
def train_stage2_classifier(X_train_s2, y_train_s2, X_test_s2, y_test_s2, num_classes, 
                             model_path=None, feature_cols_path=None, 
                             class_encoder_path=STAGE2_CLASSES_SAVE_PATH, enable_plotting=False): 
    """ Trains and evaluates the multi-class XGBoost classifier for Stage 2. """
    # ... (Full function code - NOW INCLUDES CORRECTED SAVE BLOCK from previous answer) ...
    print("\n--- Training Stage 2: Multi-Class Failure Type Classifier ---")    
    report_dict_s2 = {} ; model_s2 = None ; le = None 
    if X_train_s2.empty or y_train_s2.empty: print("Error: Training data empty."); return model_s2, report_dict_s2 
    print("Calculating sample weights..."); sample_weights_train = None
    try: sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train_s2); print(f"Sample weights calculated.")
    except Exception as e: print(f"Warning: Could not calc weights: {e}.")
    model_s2 = xgb.XGBClassifier(objective='multi:softprob', num_class=num_classes, eval_metric='mlogloss', use_label_encoder=False, n_estimators=250, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, gamma=0.1, random_state=42, n_jobs=-1)
    start_train_s2_time = datetime.now(); eval_set_s2 = [(X_test_s2, y_test_s2)]
    print("Fitting Stage 2 model...")
    try: model_s2.fit(X_train_s2, y_train_s2, sample_weight=sample_weights_train, early_stopping_rounds=30, eval_set=eval_set_s2, verbose=False)                     
    except Exception as e: print(f"Fit Error: {e}"); return None, {} 
    print(f"Training complete. Time: {(datetime.now() - start_train_s2_time).total_seconds():.2f}s")
    try: print(f"Best iteration: {model_s2.best_iteration}")
    except: print("Best iteration info unavailable.")
    print("\n--- Evaluating Stage 2 Model ---"); y_pred_s2 = None; class_names = [f'Class_{i}' for i in range(num_classes)]; 
    if class_encoder_path and os.path.exists(class_encoder_path):
        try: le = joblib.load(class_encoder_path); class_names = list(le.classes_)
        except Exception as e: print(f"Warning: Couldn't load LE: {e}")
    try: y_pred_s2 = model_s2.predict(X_test_s2); y_pred_proba_s2 = model_s2.predict_proba(X_test_s2)
    except Exception as e: print(f"Prediction Error: {e}"); return model_s2, {} 
    print("\nClassification Report (Stage 2):"); 
    try: print(classification_report(y_test_s2, y_pred_s2, target_names=class_names, zero_division=0))
    except Exception as e: print(f"Report Error: {e}"); print(classification_report(y_test_s2, y_pred_s2, zero_division=0))
    print("\nConfusion Matrix (Stage 2):"); unique_labels = np.unique(np.concatenate((y_test_s2, y_pred_s2)))
    cm_s2 = None
    try: labels_for_cm = le.transform(class_names) if le else sorted(unique_labels); cm_s2 = confusion_matrix(y_test_s2, y_pred_s2, labels=labels_for_cm); print(cm_s2)
    except Exception as e: print(f"CM Error: {e}")
    if enable_plotting and cm_s2 is not None and cm_s2.size > 0: # Plotting logic...
        try: disp_labels = class_names; fig_cm_s2, ax_cm_s2 = plt.subplots(figsize=(10, 8)); disp_cm_s2 = ConfusionMatrixDisplay(cm_s2, display_labels=disp_labels); disp_cm_s2.plot(ax=ax_cm_s2, cmap='viridis', xticks_rotation='vertical'); plt.title('Stage 2 CM'); plt.tight_layout(); plt.show()
        except Exception as e_plot: print(f"CM Plot Error: {e_plot}")
    accuracy_s2 = accuracy_score(y_test_s2, y_pred_s2); print(f"\nOverall Stage 2 Accuracy: {accuracy_s2:.2%}")
    final_feature_cols = list(X_train_s2.columns) 
    # --- CORRECTED SAVE BLOCK (ensure_dir defined globally/outside) ---
    try:
        if model_path:
            ensure_dir(model_path) # Call global function
            joblib.dump(model_s2, model_path)
            print(f"\nS2 Model saved: {model_path}")
        if feature_cols_path:
            ensure_dir(feature_cols_path) # Call global function
            joblib.dump(final_feature_cols, feature_cols_path) 
            print(f"S2 Features saved: {feature_cols_path}")
        if class_encoder_path: 
             if le is not None: 
                 ensure_dir(class_encoder_path) # Call global function
                 joblib.dump(le, class_encoder_path) 
                 print(f"S2 Label Encoder saved: {class_encoder_path}")
             else: print("Warning: LE unavailable.")
    except Exception as e: 
        print(f"Save Error: {e}")
    # --- End Corrected Save Block ---
    try: report_dict_s2 = classification_report(y_test_s2, y_pred_s2, target_names=class_names, zero_division=0, output_dict=True)
    except Exception as e: print(f"Report Dict Error: {e}"); report_dict_s2 = {}
    return model_s2, report_dict_s2

# --- FUNCTION to Save Results ---
def save_stage2_results(filepath, results_data):
    """Appends Stage 2 grid search results dictionary to a CSV file."""
    print(f"Writing Stage 2 results to {filepath}...")
    try:
        file_exists = os.path.isfile(filepath)
        # Open in append mode ('a')
        with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(results_data.keys()) 
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header only if file is newly created (or empty)
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writeheader()  
            writer.writerow(results_data)
        print("Results written successfully.")
    except IOError as e: # Catch specific I/O errors
         print(f"I/O Error writing results to CSV: {e}")
    except Exception as e: 
        print(f"General Error writing results to CSV: {e}")

# --- Helper function to ensure directory exists --- (Moved outside train function)
def ensure_dir(file_path):
    """Creates the directory for a file path if it doesn't exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory): 
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)

# --- Main Execution for Stage 2 ---
if __name__ == "__main__":

    # --- Argument Parser ---
    parser_s2 = argparse.ArgumentParser(description='Run Stage 2 Failure Type Classification.')
    parser_s2.add_argument('-w', '--window', type=int, default=DEFAULT_FEATURE_WINDOW_HOURS_S2, help=f'Feature window size (W2) in hours (default: {DEFAULT_FEATURE_WINDOW_HOURS_S2})')
    parser_s2.add_argument('-H', '--horizon', type=int, default=DEFAULT_PREDICTION_HORIZON_HOURS_S2, help=f'Prediction horizon (H2) for target mapping (default: {DEFAULT_PREDICTION_HORIZON_HOURS_S2})')
    parser_s2.add_argument('--plot', action='store_true', help='Enable plotting.')
    # --- REMOVED --fft flag ---
    args_s2 = parser_s2.parse_args()

    FEATURE_WINDOW_HOURS_S2 = args_s2.window 
    PREDICTION_HORIZON_HOURS_S2 = args_s2.horizon 
    ENABLE_PLOTTING_S2 = args_s2.plot
    # --- End Argument Parser ---

    print("--- Running Stage 2: Training Failure Type Classifier ---")
    print(f"--- Using W2={FEATURE_WINDOW_HOURS_S2}h, H2={PREDICTION_HORIZON_HOURS_S2}h ---") 
    print(f"Using Stage 1 predictions from: {STAGE1_PREDICTIONS_FILE}")
    print("*** FFT Calculation is SKIPPED ***") # Always skipped in this version
    
    # Initialize results
    run_results = { 'Timestamp_Run': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'W_Hours_S2': FEATURE_WINDOW_HOURS_S2, 'H_Hours_S2': PREDICTION_HORIZON_HOURS_S2, 'FFT_Included': False, 'Accuracy_S2': -1.0, 'F1_Macro_S2': -1.0, 'Recall_Macro_S2': -1.0, 'Precision_Macro_S2': -1.0 }

    # 1. Load Stage 1 Predictions
    try: stage1_preds_df = pd.read_csv(STAGE1_PREDICTIONS_FILE, index_col='Timestamp', parse_dates=True); #... checks ...
    except Exception as e: print(f"Load S1 Preds Error: {e}"); save_stage2_results(STAGE2_RESULTS_FILE, run_results); exit() 

    # 2. Apply Bayesian Filter 
    print("\nApplying Bayesian filter..."); stage1_preds_with_bayes = None
    try: stage1_preds_with_bayes = apply_bayesian_filter(stage1_preds_df, proba_col_name='xgb_proba', belief_alpha=BAYES_ALPHA, belief_threshold=BAYES_THRESHOLD, consecutive_steps=BAYES_STEPS); # ... check empty ...
    except Exception as e: print(f"Bayes Filter Error: {e}"); save_stage2_results(STAGE2_RESULTS_FILE, run_results); exit()
    risky_predictions_df = stage1_preds_with_bayes[stage1_preds_with_bayes['bayesian_alarm'] == 1]; print(f"\nFiltered {len(risky_predictions_df)} instances via Bayes alarm.")
    if risky_predictions_df.empty: print("No instances selected."); save_stage2_results(STAGE2_RESULTS_FILE, run_results); exit()

    # 3. Load Full Data and Create Base + Dynamic Features (Using W2)
    print(f"\nReloading data & creating base/dynamic features (W={FEATURE_WINDOW_HOURS_S2}h)...")
    sensor_df, failure_df, equipment_df, maintenance_df = load_data(data_dir=DATA_DIR)
    if sensor_df is None: print("Load data failed."); save_stage2_results(STAGE2_RESULTS_FILE, run_results); exit()
    try: # Prep merged_df ...
        if isinstance(sensor_df.index, pd.DatetimeIndex): sensor_df.reset_index(inplace=True) 
        if equipment_df is not None and not equipment_df.empty: merged_df = pd.merge(sensor_df, equipment_df, on='Machine_ID', how='left')
        else: merged_df = sensor_df 
        if 'Timestamp' in merged_df.columns: merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp']); merged_df = merged_df.set_index('Timestamp').sort_index()
        else: raise ValueError("'Timestamp' missing/lost.")
    except Exception as e: print(f"Data Prep Error: {e}"); save_stage2_results(STAGE2_RESULTS_FILE, run_results); exit()
    try: # Create features with W2 (using the SIMPLIFIED create_features from predictive_model.py)
        all_features_df = create_features(merged_df.copy(), window_size_hrs=FEATURE_WINDOW_HOURS_S2) 
    except Exception as e: print(f"Feature Creation Error: {e}"); save_stage2_results(STAGE2_RESULTS_FILE, run_results); exit()
    if all_features_df.empty: print("Feature creation failed."); save_stage2_results(STAGE2_RESULTS_FILE, run_results); exit()

    # --- FFT Features are NOT added in this version ---
        
    print(f"Final features shape for S2 prep: {all_features_df.shape}")

    # 4. Prepare Data for Stage 2 Training (Filter, Map Target H2, Encode, Split)
    print("\n--- Preparing Data for Stage 2 Classifier ---")
    X_train_s2, y_train_s2, X_test_s2, y_test_s2 = None, None, None, None 
    le = None ; final_feature_cols_s2 = [] 
    try: 
        # --- >>> CORRECTED: Use create_target_stage2 function <<<---
        stage2_data_with_target, le = create_target_stage2(
            all_features_df, # Pass the features created with W2 (NO FFT)
            failure_df, 
            prediction_horizon_hrs_s2=PREDICTION_HORIZON_HOURS_S2, # Pass H2
            stage1_risky_df=risky_predictions_df # Pass the filtered predictions
        )
        if stage2_data_with_target is None or le is None: 
             raise ValueError("Failed to create Stage 2 target data.")
             
        # --- >>> CORRECTED: Drop duplicates AFTER creating target, BEFORE split <<<---
        print(f"Shape before dropping duplicates: {stage2_data_with_target.shape}")
        # Drop based on index (Timestamp) keeping the first occurrence
        stage2_data_final = stage2_data_with_target[~stage2_data_with_target.index.duplicated(keep='first')]
        print(f"Shape after dropping duplicates: {stage2_data_final.shape}")
             
        # --- Select features and split ---
        exclude_cols = ['Machine_ID', 'Timestamp', 'Actual_Failure_Type', 'Target_Encoded', 'Failure_Within_H1_Actual', 'belief', 'consecutive_high_belief', 'bayesian_alarm'] 
        # No FFT cols to exclude in this version
        available_feature_cols = [col for col in stage2_data_final.columns if col not in exclude_cols]
        if not available_feature_cols: raise ValueError("No features available.")
        final_feature_cols_s2 = available_feature_cols # Store features used
        print(f"Using {len(final_feature_cols_s2)} features for Stage 2.")
        X_stage2 = stage2_data_final[final_feature_cols_s2]; y_stage2 = stage2_data_final['Target_Encoded'] 
        
        print(f"Stage 2 Feature shape: {X_stage2.shape}, Target shape: {y_stage2.shape}")
        
        X_train_s2, X_test_s2, y_train_s2, y_test_s2 = train_test_split(X_stage2, y_stage2, test_size=STAGE2_TEST_SIZE, random_state=42, stratify=y_stage2)
        print(f"Stage 2 Train/Test split: {len(X_train_s2)} / {len(X_test_s2)}")
        
    except Exception as e: print(f"Error preparing data for Stage 2: {e}"); save_stage2_results(STAGE2_RESULTS_FILE, run_results); exit()

    # 5. Train and Evaluate Multi-Class Model 
    # (Renumbered step, was 6 before)
    model_s2, report_s2_dict = None, {}
    if 'X_train_s2' in locals() and X_train_s2 is not None and not X_train_s2.empty: 
        num_classes = len(le.classes_) if le is not None else y_stage2.nunique(); 
        if num_classes < 2: print("Error: < 2 classes."); exit()
        print(f"\nNumber of classes: {num_classes}")
        if not os.path.exists(STAGE2_CLASSES_SAVE_PATH): print(f"Error: LE file missing."); exit()
        
        # Define temp paths using W2 and H2, mark as nofft
        fft_suffix = "nofft" 
        temp_model_path = STAGE2_MODEL_SAVE_PATH_TEMPLATE.format(w=FEATURE_WINDOW_HOURS_S2, h=PREDICTION_HORIZON_HOURS_S2, fft=fft_suffix)
        temp_features_path = STAGE2_FEATURES_SAVE_PATH_TEMPLATE.format(w=FEATURE_WINDOW_HOURS_S2, h=PREDICTION_HORIZON_HOURS_S2, fft=fft_suffix)
        
        model_s2, report_s2_dict = train_stage2_classifier(
            X_train_s2, y_train_s2, X_test_s2, y_test_s2, num_classes, 
            model_path=temp_model_path, 
            feature_cols_path=temp_features_path, 
            class_encoder_path=STAGE2_CLASSES_SAVE_PATH, 
            enable_plotting=ENABLE_PLOTTING_S2 
        )
        if model_s2 is None: print("Stage 2 model training failed.")
             
    else: print("Stage 2 data prep failed.")

    # 6. Record Results (Renumbered step, was 7 before)
    if report_s2_dict and isinstance(report_s2_dict, dict):
         run_results['Accuracy_S2']        = round(report_s2_dict.get('accuracy' ,                      -1.0), 4)
         run_results['F1_Macro_S2']        = round(report_s2_dict.get('macro avg', {}).get('f1-score' , -1.0), 4)
         run_results['Recall_Macro_S2']    = round(report_s2_dict.get('macro avg', {}).get('recall'   , -1.0), 4)
         run_results['Precision_Macro_S2'] = round(report_s2_dict.get('macro avg', {}).get('precision', -1.0), 4)
         
    # Save results for this W/H combination (NO FFT)
    save_stage2_results(STAGE2_RESULTS_FILE, run_results)

    print("\n--- Stage 2 Finished ---")
