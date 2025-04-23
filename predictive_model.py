# predictive_model.py (FINAL VERSION - Incorporating all fixes)

import matplotlib; matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve, 
    auc,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    f1_score 
)
from scipy.stats import linregress 
import xgboost as xgb 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# import seaborn as sns # Not used directly here
import joblib 
import os
import argparse 
import csv 
import warnings

# --- Configure Warnings ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) 
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning) 

# --- Configuration ---
DATA_DIR = 'projects/uploads/' # Or your correct data path
# DATA_DIR = 'data/10apr/'
SENSOR_FILE = os.path.join(DATA_DIR, 'sensor_data.csv')
FAILURE_FILE = os.path.join(DATA_DIR, 'failure_logs.csv')
EQUIPMENT_FILE = os.path.join(DATA_DIR, 'equipment_usage.csv')
MAINTENANCE_FILE = os.path.join(DATA_DIR, 'maintenance_history.csv') 
RESULTS_FILE = 'projects/script_result/stage1_grid_search_WH_results.csv' # Results file for Stage 1 Grid Search
DEFAULT_MODEL_SAVE_PATH = 'projects/script_result/predictive_model_xgb_s1.joblib' 
DEFAULT_FEATURE_COLS_SAVE_PATH = 'projects/script_result/feature_columns_xgb_s1.joblib' 
STAGE1_PREDICTIONS_OUTPUT_FILE = 'projects/script_result/stage1_predictions.csv' # Output for Stage 2
# Default values for window and horizon (Stage 1)
DEFAULT_FEATURE_WINDOW_HOURS = 24  # Example optimal W1
DEFAULT_PREDICTION_HORIZON_HOURS = 48 # Example optimal H1
DEFAULT_TEST_SET_SIZE = 0.3     
DEFAULT_BAYES_ALPHA = 0.7       # Default Bayes params
DEFAULT_BAYES_THRESHOLD = 0.75
DEFAULT_BAYES_STEPS = 2

# --- Helper function to ensure directory exists --- 
def ensure_dir(file_path):
    """Creates the directory for a file path if it doesn't exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory): 
        try: 
             print(f"Creating directory: {directory}")
             os.makedirs(directory, exist_ok=True)
        except OSError as e: print(f"Warn: Could not create dir {directory}: {e}")

# --- Core Functions ---

def load_data(sensor_file=SENSOR_FILE, failure_file=FAILURE_FILE, 
              equipment_file=EQUIPMENT_FILE, maintenance_file=MAINTENANCE_FILE,
              data_dir=DATA_DIR):
    """Loads the generated CSV files."""
    # ... (Exact same working version from your code) ...
    print("--- Loading Data ---")
    sensor_file=os.path.join(data_dir, os.path.basename(sensor_file)); failure_file=os.path.join(data_dir, os.path.basename(failure_file)); equipment_file=os.path.join(data_dir, os.path.basename(equipment_file)); maintenance_file=os.path.join(data_dir, os.path.basename(maintenance_file))
    loaded_data={}; 
    try:
        if not os.path.exists(data_dir): raise FileNotFoundError(f"Dir not found: {data_dir}")
        loaded_data['sensor']=pd.read_csv(sensor_file, parse_dates=['Timestamp']); 
        if loaded_data['sensor'].empty: raise ValueError("Sensor empty.")
        if os.path.exists(failure_file) and os.path.getsize(failure_file) > 0: loaded_data['failure']=pd.read_csv(failure_file, parse_dates=['Timestamp'])
        else: print(f"Warn: Failure log '{failure_file}' empty."); loaded_data['failure'] = pd.DataFrame(columns=['Machine_ID', 'Timestamp', 'Failure_Type']); loaded_data['failure']['Timestamp'] = pd.to_datetime(loaded_data['failure']['Timestamp'])
        if os.path.exists(equipment_file) and os.path.getsize(equipment_file) > 0: loaded_data['equipment']=pd.read_csv(equipment_file)
        else: print(f"Warn: Equipment file '{equipment_file}' empty."); loaded_data['equipment'] = pd.DataFrame(columns=['Machine_ID', 'Equipment_Age (Years)', 'Usage_Cycles'])
        if os.path.exists(maintenance_file) and os.path.getsize(maintenance_file) > 0: loaded_data['maintenance']=pd.read_csv(maintenance_file, parse_dates=['Timestamp'])
        else: print(f"Warn: Maintenance log '{maintenance_file}' empty."); loaded_data['maintenance'] = pd.DataFrame(columns=['Machine_ID', 'Timestamp', 'Maintenance_Action']); loaded_data['maintenance']['Timestamp'] = pd.to_datetime(loaded_data['maintenance']['Timestamp'])
        print(f"Loaded sensor: {loaded_data['sensor'].shape}"); print(f"Loaded failure: {loaded_data['failure'].shape}"); print(f"Loaded equipment: {loaded_data['equipment'].shape}"); print(f"Loaded maintenance: {loaded_data['maintenance'].shape}")
        return loaded_data['sensor'], loaded_data['failure'], loaded_data['equipment'], loaded_data['maintenance']
    except FileNotFoundError as e: print(f"Error loading file: {e}."); return None, None, None, None
    except Exception as e: print(f"Error loading data: {e}"); return None, None, None, None

# --- Helper function for rolling slope --- (DEFINED OUTSIDE create_features)
def calculate_rolling_slope(series):
    """Calculates the slope of a linear regression on the series."""
    y = series.dropna()
    # Need at least 3 points for a somewhat reliable slope, 2 for calculation
    if len(y) < 2: 
        return np.nan 
    x = np.arange(len(y))
    try: 
        # Use polyfit for just the slope (often faster than full linregress)
        # Degree 1 polynomial: coefficients are [slope, intercept]
        slope = np.polyfit(x, y, 1)[0] 
        # Handle potential NaN slope if polyfit fails on weird data
        return slope if not np.isnan(slope) else 0.0 
    except (np.linalg.LinAlgError, ValueError): # Catch potential errors
        return np.nan # Or 0.0 ? NaN might be better signal of failure

def create_features(df, window_size_hrs, sensor_cols=['Temperature', 'Vibration', 'Pressure', 'Current', 'AFR', 'RPM']):
    """ Engineers features: rolling stats, skew, kurt, diff, slope, std_diff, corr. (Corrected Slope Application) """
    print(f"\n--- Creating Features (W={window_size_hrs}h, No FFT) ---")
    if not isinstance(df.index, pd.DatetimeIndex): df = df.set_index('Timestamp').sort_index()
    if df.empty: print("Error: Input empty."); return pd.DataFrame() 
    
    df_out = df.copy() # Work on a copy

    # --- Robust Frequency Inference & Window Periods ---
    # ... (Use the last CORRECT robust frequency/window period code) ...
    freq = None; window_periods = 0; # Init
    try: freq = pd.infer_freq(df_out.index[:min(len(df_out.index), 20000)]); 
    except Exception: pass
    if freq: freq_offset_validation = pd.tseries.frequencies.to_offset(freq); 
    if freq is None and len(df_out.index) > 1: 
        print("Attempting manual frequency inference from median difference...")
        # --- Start TRY for Manual Inference ---
        try: 
            unique_sorted_times = df_out.index.unique().sort_values()
            if len(unique_sorted_times) > 1:
                time_diffs = np.diff(unique_sorted_times)
                positive_diffs = time_diffs[time_diffs > np.timedelta64(0)]                 
                if len(positive_diffs) > 0:
                    median_diff_ns = np.median(positive_diffs).astype('timedelta64[ns]').astype(np.int64)                     
                    if median_diff_ns > 0:
                        inferred_td = pd.Timedelta(nanoseconds=median_diff_ns) 
                        freq_offset = pd.tseries.frequencies.to_offset(inferred_td)
                        if freq_offset: 
                            freq = freq_offset # Assign freq ONLY if successful
                            print(f"Inferred frequency manually: {freq}")
                        else:
                            print("Warn: Manual inference - could not convert timedelta to offset.")
                    else:
                        print("Warn: Manual inference - median difference is non-positive.")
                else:
                    print("Warn: Manual inference - no positive time differences found.")
            else:
                print("Warn: Manual inference - not enough unique timestamps.")                 
        # --- CORRECTED INDENTATION for except ---
        except Exception as e: 
            print(f"Warn: Manual frequency inference failed with error: {e}")          
    if not freq: fallback_freq_str = '1800s'; freq = pd.tseries.frequencies.to_offset(fallback_freq_str); print(f"Warn: Falling back freq: {freq}")
    try: # Calculate periods
        if not isinstance(freq, pd.offsets.BaseOffset): freq = pd.tseries.frequencies.to_offset(freq)
        freq_timedelta = pd.Timedelta(freq); 
        if freq_timedelta.total_seconds() <= 0: raise ValueError("Freq Timedelta <= 0")
        window_periods = int(timedelta(hours=window_size_hrs) / freq_timedelta)
    except Exception as e: print(f"Error calc periods: {e}"); window_periods = int(timedelta(hours=window_size_hrs) / timedelta(seconds=1800)); print(f"Using default periods: {window_periods}")
    if window_periods <= 0: window_periods = 2;
    if window_periods < 2 : print(f"Warn: Window periods {window_periods}<2."); window_periods = 2
    min_periods_base = max(1, window_periods // 2); min_periods_slope = max(3, min_periods_base); min_periods_skew_kurt = max(4, min_periods_base)
    print(f"Using window periods: {window_periods}")
    
    # --- Feature Calculation ---
    print("Calculating rolling features...")
    start_time_feat = datetime.now() 
    
    # Ensure Machine_ID is a column for grouping
    if 'Machine_ID' not in df_out.columns: df_out_grouped = df_out.reset_index() if df_out.index.name == 'Machine_ID' else df_out;
    else: df_out_grouped = df_out 
    if df_out_grouped.index.name == 'Machine_ID': df_out_grouped = df_out_grouped.reset_index()
    if not isinstance(df_out_grouped.index, pd.DatetimeIndex):
         if 'Timestamp' in df_out_grouped.columns: df_out_grouped = df_out_grouped.set_index('Timestamp').sort_index()
         else: print("Error: Timestamp index missing."); return pd.DataFrame()

    # Group once for efficiency
    grouped_data = df_out_grouped.groupby('Machine_ID')

    # Calculate features per group and assign back
    for col in sensor_cols:
        if col not in df_out.columns: continue
        print(f"  Calculating features for: {col}...") 

        # --- Shift within groups using transform ---
        shifted_col = grouped_data[col].transform(lambda x: x.shift(1))
        
        # --- Calculate rolling stats on the shifted series (aligned with df_out index) ---
        rolling_base = shifted_col.rolling(window=window_periods, min_periods=min_periods_base)
        rolling_dist = shifted_col.rolling(window=window_periods, min_periods=min_periods_skew_kurt)
        rolling_slope_calc = shifted_col.rolling(window=window_periods, min_periods=min_periods_slope) 
        
        # Assign basic stats directly 
        df_out[f'{col}_rol_mean_{window_size_hrs}h'] = rolling_base.mean() 
        df_out[f'{col}_rol_std_{window_size_hrs}h']  = rolling_base.std()   
        df_out[f'{col}_rol_min_{window_size_hrs}h']  = rolling_base.min()   
        df_out[f'{col}_rol_max_{window_size_hrs}h']  = rolling_base.max()   
        df_out[f'{col}_rol_skew_{window_size_hrs}h'] = rolling_dist.skew()
        df_out[f'{col}_rol_kurt_{window_size_hrs}h'] = rolling_dist.kurt()
        
        # Assign diff (can still use transform here on original group)
        df_out[f'{col}_diff_1step'] = grouped_data[col].transform(lambda x: x.shift(1).diff(periods=1))
        
        # --- CORRECTED: Assign rolling slope using apply ---
        # Apply the globally defined function
        df_out[f'{col}_rol_slope_{window_size_hrs}h'] = rolling_slope_calc.apply(calculate_rolling_slope, raw=False) # Use raw=False for Series input

    # --- Calculate derived features ---
    print("  Calculating derived features...")
    for col in sensor_cols:
         if col not in df_out.columns: continue
         std_col_name = f'{col}_rol_std_{window_size_hrs}h' 
         if std_col_name in df_out.columns:
              # Use transform for diff on the std column
              df_out[f'{col}_rol_std_diff_1step'] = df_out.groupby('Machine_ID')[std_col_name].transform(lambda x: x.diff(periods=1))
              
    if 'Vibration' in sensor_cols and 'Current' in sensor_cols: # Correlation
        print("  Calculating correlation...")
        shifted_vib = df_out.groupby('Machine_ID')['Vibration'].transform(lambda x: x.shift(1)) # Use transform
        shifted_cur = df_out.groupby('Machine_ID')['Current'].transform(lambda x: x.shift(1))   # Use transform
        # Calculate rolling correlation on shifted Series
        df_out[f'Vib_Curr_corr_{window_size_hrs}h'] = shifted_vib.rolling(window=window_periods, min_periods=max(2, min_periods_base)).corr(shifted_cur)

    print(f"\n  Feature calculation complete. Time: {(datetime.now() - start_time_feat).total_seconds():.2f}s")
    print("  Handling NaNs..."); df_out = df_out.fillna(method='bfill').fillna(0); 
    
    # --- Corrected Static Feature Addition ---
    static_cols = ['Equipment_Age (Years)', 'Usage_Cycles']; 
    valid_static_cols = [scol for scol in df.columns if scol in static_cols] # Check original df
    if valid_static_cols: 
        print("  Adding static features..."); 
        try: 
            # Use map if index is unique after potential reset/set_index earlier
            if df_out.index.is_unique and 'Machine_ID' in df_out.columns:
                 static_map_age = df.drop_duplicates(subset=['Machine_ID']).set_index('Machine_ID')['Equipment_Age (Years)']
                 static_map_cyc = df.drop_duplicates(subset=['Machine_ID']).set_index('Machine_ID')['Usage_Cycles']
                 df_out['Equipment_Age (Years)'] = df_out['Machine_ID'].map(static_map_age)
                 df_out['Usage_Cycles'] = df_out['Machine_ID'].map(static_map_cyc)
            else: # Fallback to merge if index not unique or Machine_ID issue
                 print("    Using merge for static features due to index/column state...")
                 df_static = df[['Machine_ID'] + valid_static_cols].drop_duplicates(subset=['Machine_ID'], keep='first')
                 df_out_to_merge = df_out.reset_index() # Reset index for merge
                 df_out_merged = pd.merge(df_out_to_merge, df_static, on='Machine_ID', how='left', suffixes=('', '_y')) 
                 cols_to_drop = [col for col in df_out_merged.columns if '_y' in col]; df_out_merged.drop(columns=cols_to_drop, inplace=True)
                 if 'Timestamp' in df_out_merged.columns: df_out = df_out_merged.set_index('Timestamp').sort_index()
                 else: df_out = df_out_merged
            print(f"Static features added. Shape: {df_out.shape}")
        except Exception as e: print(f"Warning: Error adding static features: {e}")
    else: print("Warn: Static features not found.")
    
    print("Feature creation complete.")
    return df_out
    
    # Calculate basic stats + slope + diff
    for col in sensor_cols:
        if col not in df_out.columns: continue
        print(f"  Calculating features for: {col}...") 
        shifted_col_grouped = df_out_grouped.groupby('Machine_ID')[col].shift(1)
        rolling_base = shifted_col_grouped.rolling(window=window_periods, min_periods=min_periods_base)
        rolling_dist = shifted_col_grouped.rolling(window=window_periods, min_periods=min_periods_skew_kurt)
        rolling_slope_calc = shifted_col_grouped.rolling(window=window_periods, min_periods=min_periods_slope) 
        df_out[f'{col}_rol_mean_{window_size_hrs}h'] = rolling_base.mean(); df_out[f'{col}_rol_std_{window_size_hrs}h']  = rolling_base.std()   
        df_out[f'{col}_rol_min_{window_size_hrs}h']  = rolling_base.min(); df_out[f'{col}_rol_max_{window_size_hrs}h']  = rolling_base.max()   
        df_out[f'{col}_rol_skew_{window_size_hrs}h'] = rolling_dist.skew(); df_out[f'{col}_rol_kurt_{window_size_hrs}h'] = rolling_dist.kurt()
        df_out[f'{col}_diff_1step'] = df_out_grouped.groupby('Machine_ID')[col].transform(lambda x: x.shift(1).diff(periods=1))
        df_out[f'{col}_rol_slope_{window_size_hrs}h'] = rolling_slope_calc.apply(rolling_slope) # NO raw=True

    # --- Calculate derived features (std_diff, correlation) - CORRECTED ---
    print("  Calculating derived features...")
    for col in sensor_cols:
         if col not in df_out.columns: continue
         std_col_name = f'{col}_rol_std_{window_size_hrs}h' # Define std_col_name HERE
         if std_col_name in df_out.columns:
              df_out[f'{col}_rol_std_diff_1step'] = df_out.groupby('Machine_ID')[std_col_name].transform(lambda x: x.diff(periods=1))
              
    if 'Vibration' in sensor_cols and 'Current' in sensor_cols: # Correlation
        print("  Calculating correlation...")
        shifted_vib = df_out_grouped.groupby('Machine_ID')['Vibration'].shift(1); shifted_cur = df_out_grouped.groupby('Machine_ID')['Current'].shift(1)
        df_out[f'Vib_Curr_corr_{window_size_hrs}h'] = shifted_vib.rolling(window=window_periods, min_periods=max(2, min_periods_base)).corr(shifted_cur)

    print(f"\n  Feature calculation complete. Time: {(datetime.now() - start_time_feat).total_seconds():.2f}s")
    print("  Handling NaNs..."); df_out = df_out.fillna(method='bfill').fillna(0); 
    
    # --- Corrected Static Feature Addition ---
    static_cols = ['Equipment_Age (Years)', 'Usage_Cycles']; 
    valid_static_cols = [scol for scol in static_cols if scol in df.columns] # Check original df columns
    if valid_static_cols: 
        print("  Adding static features..."); 
        try: 
            df_static = df[['Machine_ID'] + valid_static_cols].drop_duplicates(subset=['Machine_ID'], keep='first')
            # Merge static based on Machine_ID column 
            if 'Machine_ID' not in df_out.columns: df_out_to_merge = df_out.reset_index()
            else: df_out_to_merge = df_out
            if 'Machine_ID' in df_out_to_merge.columns:
                original_index = df_out.index 
                df_out_merged = pd.merge(df_out_to_merge.reset_index(), df_static, on='Machine_ID', how='left', suffixes=('', '_y')) # Reset index before merge
                cols_to_drop = [col for col in df_out_merged.columns if '_y' in col]; df_out_merged.drop(columns=cols_to_drop, inplace=True)
                # Set index back
                if 'Timestamp' in df_out_merged.columns: df_out = df_out_merged.set_index('Timestamp').sort_index()
                else: df_out = df_out_merged; print("Warn: Timestamp missing after static merge.")
                # Check if original index was preserved
                if not df_out.index.equals(original_index) and isinstance(original_index, pd.DatetimeIndex):
                     print("Warning: Index changed during static feature merge. Attempting reindex.")
                     # Try reindexing back (might fail if duplicates were introduced/handled differently)
                     try: df_out = df_out.reindex(original_index)
                     except Exception as reindex_e: print(f"Reindex failed: {reindex_e}")
                print(f"Static features added. Shape: {df_out.shape}")
            else: print("Warn: Machine_ID missing for static merge.")
        except Exception as e: print(f"Warn: Error adding static features: {e}")
    else: print("Warn: Static features not found in original df.")
    
    print("Feature creation complete.")
    return df_out

# Patch to fix feature mismatch between train and inference.

def align_features(df_features, required_feature_list, fill_value=0.0):
    """
    Ensures that df_features contains exactly the columns in required_feature_list, in the right order.
    Missing columns are added with fill_value. Extra columns are dropped.
    """
    import numpy as np
    import pandas as pd

    df_aligned = df_features.copy()
    current_cols = set(df_aligned.columns)
    required_cols = set(required_feature_list)

    # Add missing columns
    missing = required_cols - current_cols
    for col in missing:
        df_aligned[col] = fill_value

    # Drop extra columns
    extra = current_cols - required_cols
    df_aligned = df_aligned.drop(columns=list(extra), errors='ignore')

    # Reorder to match required_feature_list
    df_aligned = df_aligned[required_feature_list]

    return df_aligned

# --- Function to Create Target ---
def create_target(df, failure_df, prediction_horizon_hrs, target_col_name='Failure_Within_H', roll_start_ts=None, roll_end_ts=None):
    """Creates the binary target variable."""
    # ... (No changes needed from previous correct version) ...
    print(f"\n--- Creating Target Variable '{target_col_name}' (H={prediction_horizon_hrs}h) ---")
    if not isinstance(df.index, pd.DatetimeIndex): print("Error: df index must be DatetimeIndex."); return df
    df_out = df.copy(); 
    if failure_df is None or failure_df.empty: print("Warning: Failure log empty."); df_out[target_col_name] = 0; return df_out
    if not df_out.index.is_monotonic_increasing: print("Warning: Sorting index."); df_out = df_out.sort_index()
    failure_df = failure_df.sort_values(by=['Machine_ID', 'Timestamp']); horizon_delta = timedelta(hours=prediction_horizon_hrs); df_out[target_col_name] = 0 
    failure_lookup = failure_df.groupby('Machine_ID')['Timestamp'].apply(np.array).to_dict(); target_indices = []; 
    print("  Assigning target labels..."); total_rows = len(df_out); processed_rows = 0; start_target_time = datetime.now()
    try: machine_id_array = df_out['Machine_ID'].values
    except KeyError: print("Error: 'Machine_ID' missing."); return df_out
    index_array = df_out.index.values 
    for i in range(total_rows): # Target Assignment Loop
        processed_rows += 1; 
        if processed_rows % 100000 == 0: elapsed = (datetime.now() - start_target_time).total_seconds(); rate = processed_rows / elapsed if elapsed > 0 else 0; print(f"    Processed {processed_rows}/{total_rows}...", end='\r')
        machine_id = machine_id_array[i]; current_time_ts = pd.Timestamp(index_array[i]); window_end = current_time_ts + horizon_delta
        if machine_id in failure_lookup:
            failure_times = failure_lookup[machine_id]; failures_in_window = failure_times[(failure_times > current_time_ts) & (failure_times <= window_end)]
            if len(failures_in_window) > 0: target_indices.append(index_array[i])                 
    print(f"\n    Finished target loop. Time: {(datetime.now() - start_target_time).total_seconds():.2f}s")
    if target_indices: # Set target
        unique_target_indices = pd.Index(target_indices).unique(); valid_indices = df_out.index.intersection(unique_target_indices)
        if not valid_indices.empty: df_out.loc[valid_indices, target_col_name] = 1; print(f"  Set {len(valid_indices)} targets.")
    print("Target created."); print(f"Target proportion:\n{df_out[target_col_name].value_counts(normalize=True)}")
    return df_out
      
# --- Function to Train XGBoost Model ---
def train_xgboost_model(features_df, target_col='Failure_Within_H', test_size=DEFAULT_TEST_SET_SIZE, enable_plotting=False, model_path=None, feature_cols_path=None): 
    """ Trains an XGBoost model (v1.x compatible) and evaluates it. """
    # ... (Full function code - Includes CORRECTED SAVE BLOCK) ...
    print(f"\n--- Training Model (XGBoost v1.x, Target: {target_col}) ---")
    model, trained_feature_cols = None, None; X_test, y_test, y_pred_proba, machine_ids_test = None, None, None, None 
    roc_auc, pr_auc = -1.0, -1.0; valid_cols = [] 
    potential_feature_cols = [col for col in features_df.columns if col not in ['Machine_ID', 'Timestamp', target_col] and not pd.api.types.is_datetime64_any_dtype(features_df[col])] 
    if not features_df.empty: # Feature Cleaning
         temp_X = features_df[potential_feature_cols].copy();
         for col in temp_X.columns:
             if temp_X[col].dtype == 'object': # Exclude non-numeric objects
                  try: pd.to_numeric(temp_X[col]); 
                  except ValueError: print(f"  Excluding object: {col}"); continue 
             if temp_X[col].nunique(dropna=False) < 2 or temp_X[col].isnull().all(): print(f"  Excluding constant/NaN: {col}"); continue # Exclude constant/NaN
             valid_cols.append(col)
    feature_cols = valid_cols
    if not feature_cols: print("Error: No valid features."); return model, feature_cols, X_test, y_test, y_pred_proba, machine_ids_test, roc_auc, pr_auc
    print(f"Using {len(feature_cols)} features.")
    X = features_df[feature_cols]; y = features_df[target_col]
    n_samples = len(X); split_point = int(n_samples * (1 - test_size)); X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]; print(f"Train/Test sizes: {len(X_train)} / {len(X_test)}")
    train_counts = y_train.value_counts(); test_counts = y_test.value_counts(); 
    if len(train_counts) < 2: print("Error: Only one class in train target."); return model, feature_cols, X_test, y_test, y_pred_proba, machine_ids_test, roc_auc, pr_auc
    machine_ids_test_array = None # Get Machine IDs
    if 'Machine_ID' in features_df.columns:
         try: machine_ids_test_array = features_df['Machine_ID'].iloc[split_point:].values; 
         except Exception as e: print(f"Error getting Machine_IDs: {e}")
    print("\nTraining XGBoost..."); neg_count = train_counts.get(0, 0); pos_count = train_counts.get(1, 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1; print(f"Scale_pos_weight: {scale_pos_weight:.2f}")
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', scale_pos_weight=scale_pos_weight, n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.7, colsample_bytree=0.7, gamma=0.2, random_state=42, n_jobs=-1)
    start_train_time = datetime.now(); eval_set = [(X_test, y_test)] 
    try: model.fit(X_train, y_train, early_stopping_rounds=30, eval_set=eval_set, verbose=False)                   
    except TypeError: print(f"Fit error. Retrying without early stopping."); model.fit(X_train, y_train, verbose=False) 
    except Exception as e: print(f"Fit error: {e}"); return model, feature_cols, X_test, y_test, y_pred_proba, machine_ids_test, roc_auc, pr_auc
    print(f"Training complete. Time: {(datetime.now() - start_train_time).total_seconds():.2f}s")
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration > 0 else model.n_estimators
    print(f"Best iteration: {best_iteration if hasattr(model, 'best_iteration') and model.best_iteration > 0 else 'N/A (Final)'}")
    print("\n--- Evaluating Model ---"); 
    try: y_pred_proba = model.predict_proba(X_test)[:, 1] 
    except Exception as e: print(f"Predict_proba error: {e}."); return model, feature_cols, X_test, y_test, y_pred_proba, machine_ids_test, roc_auc, pr_auc
    try: roc_auc = roc_auc_score(y_test, y_pred_proba)
    except Exception: print("ROC AUC calc error")
    try: precision, recall, _ = precision_recall_curve(y_test, y_pred_proba); pr_auc = auc(recall, precision)
    except Exception: print("PR AUC calc error")
    print(f"Overall ROC AUC Score: {roc_auc:.4f}"); print(f"Overall PR AUC Score: {pr_auc:.4f}")
    if enable_plotting: # Plotting logic...
        fig, axes = plt.subplots(1, 2, figsize=(16, 7)); fig.suptitle('XGBoost Evaluation (Stage 1)', fontsize=16)  #... ROC/PR plots ...
        try: RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=axes[0], name='XGBoost'); axes[0].set_title(f'ROC (AUC={roc_auc:.4f})'); axes[0].plot([0, 1], [0, 1], 'k--'); axes[0].legend(); axes[0].grid(True)
        except: axes[0].set_title('ROC (Error)')
        try: PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba, ax=axes[1], name=f'XGBoost (AUC={pr_auc:.4f})'); axes[1].set_title(f'PR Curve (AUC={pr_auc:.4f})'); axes[1].grid(True);
        except: axes[1].set_title('PR Curve (Error)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
        try: # Feature Importance ...
             booster = model.get_booster(); importances = booster.get_score(importance_type='gain'); imp_series = pd.Series(importances).sort_values(ascending=False)
             n_plot = min(15, len(feature_cols)); top_imp = imp_series.head(n_plot); plt.figure(figsize=(10, max(5, n_plot//2))); plt.barh(top_imp.index, top_imp.values); plt.xlabel('Gain'); plt.title(f'Top {n_plot} Features (S1)'); plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
             print("\nTop 15 Features (Gain):"); print(imp_series.head(15)) 
        except Exception as e: print(f"Feature Importance Plot Error: {e}")
    # --- Corrected Save Block ---
    if model_path and feature_cols_path:
        try:
            ensure_dir(model_path); joblib.dump(model, model_path)
            ensure_dir(feature_cols_path); joblib.dump(feature_cols, feature_cols_path) 
            print(f"\nModel saved: {model_path}"); print(f"Features saved: {feature_cols_path}")
        except Exception as e: print(f"Error saving model: {e}")
    return model, feature_cols, X_test, y_test, y_pred_proba, machine_ids_test_array, roc_auc, pr_auc

# --- FUNCTION to Apply Bayesian Filter ---
def apply_bayesian_filter(df_test_results, proba_col_name='xgb_proba', 
                          belief_alpha=DEFAULT_BAYES_ALPHA, 
                          belief_threshold=DEFAULT_BAYES_THRESHOLD, 
                          consecutive_steps=DEFAULT_BAYES_STEPS):
    """Applies a simple Bayesian-like filter to smooth probabilities."""
    # ... (Full function code - NO CHANGES NEEDED) ...
    print(f"\n--- Applying Bayesian Filter (alpha={belief_alpha}, th={belief_threshold}, steps={consecutive_steps}) ---")
    required_cols = ['Machine_ID', proba_col_name]; 
    if not all(col in df_test_results.columns for col in required_cols) or not isinstance(df_test_results.index, pd.DatetimeIndex): print("Error: Input DF invalid."); return pd.DataFrame() 
    df_results = df_test_results.sort_values(by=['Machine_ID', 'Timestamp']).copy(); beliefs = []; counts = []; last_m = None; cur_b = 0.0; cur_c = 0; n = len(df_results); start_bayes_time = datetime.now()
    for i, (idx, row) in enumerate(df_results.iterrows()): 
        if i % 100000 == 0: print(f"  Filter progress {i}/{n}...", end='\r')
        m = row['Machine_ID']; p = row[proba_col_name] 
        if m != last_m: cur_b = 0.0 ; cur_c = 0; last_m = m
        cur_b = belief_alpha * cur_b + (1 - belief_alpha) * p; beliefs.append(cur_b)
        if cur_b >= belief_threshold: cur_c += 1
        else: cur_c = 0 
        counts.append(cur_c)
    # Correct timer variable name if needed (use start_bayes_time)
    print(f"\n  Filter done. Time: {(datetime.now() - start_bayes_time).total_seconds():.2f}s") 
    df_results['belief'] = beliefs; df_results['consecutive_high_belief'] = counts
    df_results['bayesian_alarm'] = (df_results['consecutive_high_belief'] >= consecutive_steps).astype(int)
    print(f"Bayes Alarm dist:\n{df_results['bayesian_alarm'].value_counts()}")
    return df_results

# --- FUNCTION to Save Grid Search Results ---
def save_results_to_csv(filepath, results_data):
    """Appends results dictionary to a CSV file using simple file writing."""
    print(f"\nAttempting to write results to {filepath}...")
    try:
        ensure_dir(filepath) # Ensure directory exists
        file_exists = os.path.isfile(filepath)
        # Define header order explicitly for consistency
        fieldnames = ['Timestamp_Run', 'W_Hours', 'H_Hours', 'ROC_AUC', 'PR_AUC', 
                      'Max_F1', 'Optimal_Threshold', 'Precision_at_Max_F1', 
                      'Recall_at_Max_F1', 'Bayes_F1', 'Bayes_Precision', 'Bayes_Recall']
        
        # Construct the data line string
        # Use get(key, default_value) to handle potentially missing keys in results_data
        data_values = [str(results_data.get(key, 'ERROR')) for key in fieldnames]
        line_to_write = ",".join(data_values) + "\n"

        # Open in append mode ('a') with utf-8 encoding
        with open(filepath, 'a', encoding='utf-8') as f:
            # Write header only if file is new or empty
            if not file_exists or os.path.getsize(filepath) == 0:
                header_line = ",".join(fieldnames) + "\n"
                f.write(header_line)
            # Write the data line
            f.write(line_to_write)
        print("Results written successfully.")
        
    except IOError as e: # Catch specific I/O errors
         print(f"I/O Error writing results to CSV: {e}")
    except Exception as e: 
        print(f"General Error writing results to CSV: {e}")

# --- FUNCTION to Calculate Metrics at Threshold ---
def calculate_metrics_at_threshold(y_true, y_pred_proba, threshold):
    """Calculates Precision, Recall, F1 for a given threshold."""
    # ... (Same as before) ...
    if y_pred_proba is None or y_true is None: return 0, 0, 0, 0 
    y_pred_label = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_label, labels=[0,1]) 
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0], 0, 0, 0) if cm.shape==(1,1) and y_true.iloc[0]==0 else (0, 0, cm[0,0], 0) if cm.shape==(1,1) and y_true.iloc[0]==1 else (0,0,0,0)
    alerts = tp + fp; actual_positives = tp + fn
    precision = tp / alerts if alerts > 0 else 0
    recall = tp / actual_positives if actual_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9) 
    return precision, recall, f1, alerts

# --- Main Orchestration Function (CORRECTED DEFINITION) ---
def run_pipeline(feature_window_hrs, prediction_horizon_hrs, 
                 test_size=DEFAULT_TEST_SET_SIZE, enable_plotting=False,
                 results_filepath=RESULTS_FILE,
                 model_save_path=None, 
                 feature_cols_save_path=None, # <<<--- CORRECTED: Added this parameter
                 stage1_pred_output_path=STAGE1_PREDICTIONS_OUTPUT_FILE): 
    """Runs the full pipeline for Stage 1."""
    
    print("="*70); print(f" Starting Run: W={feature_window_hrs}h, H={prediction_horizon_hrs}h "); print("="*70)
    results = {'Timestamp_Run': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'W_Hours': feature_window_hrs, 'H_Hours': prediction_horizon_hrs,'ROC_AUC': -1.0, 'PR_AUC': -1.0, 'Max_F1': 0.0, 'Optimal_Threshold': 0.5,'Precision_at_Max_F1': -1.0, 'Recall_at_Max_F1': -1.0,'Bayes_F1': -1.0, 'Bayes_Precision': -1.0, 'Bayes_Recall': -1.0 }

    try: # Wrap major steps in try-except
        # 1. Load Data
        sensor_df, failure_df, equipment_df, maintenance_df = load_data()
        if sensor_df is None: raise ValueError("Data loading failed.")
        
        # 2. Prepare Data
        print("Preparing data..."); merged_df = pd.DataFrame() 
        if isinstance(sensor_df.index, pd.DatetimeIndex): sensor_df.reset_index(inplace=True) 
        if 'Timestamp' not in sensor_df.columns: raise ValueError("'Timestamp' missing.")
        if equipment_df is not None and not equipment_df.empty: merged_df = pd.merge(sensor_df, equipment_df, on='Machine_ID', how='left')
        else: merged_df = sensor_df.copy() 
        if 'Timestamp' in merged_df.columns: merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp']); merged_df = merged_df.set_index('Timestamp').sort_index()
        else: raise ValueError("'Timestamp' lost.")
        if merged_df.index.duplicated().any(): print("WARNING: Duplicates after merge!")

        # 3. Create Features
        features_df = create_features(merged_df.copy(), window_size_hrs=feature_window_hrs) 
        if features_df.empty: raise ValueError("Feature creation failed.")

        # 4. Create Target 
        final_df = pd.DataFrame()
        if failure_df is not None: 
             final_df = create_target(features_df, failure_df, prediction_horizon_hrs=prediction_horizon_hrs)
             if final_df.empty: raise ValueError("Target creation failed.")
        else: final_df = features_df; final_df['Failure_Within_H'] = 0 
        if final_df.empty: raise ValueError("Data empty after target.")

        # --- Apply final deduplication ---
        print(f"Shape before final dedupe: {final_df.shape}")
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        print(f"Shape after final dedupe: {final_df.shape}")
        if final_df.empty: raise ValueError("Empty after dedupe.")

        # 5. Train & Evaluate Model 
        model, trained_feature_cols = None, None; X_test, y_test, y_pred_proba_test, machine_ids_test = None, None, None, None 
        roc_auc_score_val, pr_auc_score_val = -1.0, -1.0 
        model_trained_successfully = False
                 
        if 'Failure_Within_H' in final_df.columns and final_df['Failure_Within_H'].nunique() > 1:
              # --- Pass feature_cols_save_path correctly ---
              model, trained_feature_cols, X_test, y_test, y_pred_proba_test, machine_ids_test, roc_auc, pr_auc = train_xgboost_model(
                  final_df, target_col='Failure_Within_H', test_size=test_size, enable_plotting=enable_plotting,
                  model_path=model_save_path, 
                  feature_cols_path=feature_cols_save_path # Pass the argument here
              )
              if model is not None and y_pred_proba_test is not None: 
                   model_trained_successfully = True
                   results['ROC_AUC'] = round(roc_auc, 4); results['PR_AUC'] = round(pr_auc, 4)
        else: print("\nSkipping model training."); model_trained_successfully = False

        # 6. Threshold Analysis & F1 Optimization 
        optimal_threshold_f1 = 0.5; max_f1_score = 0.0; precision_at_max_f1, recall_at_max_f1 = -1.0, -1.0
        if model_trained_successfully: 
            print("\n--- Detailed Threshold Analysis (Stage 1) ---")
            try: # F1 Optimization... 
                precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba_test)
                f1_scores = 2*(precision*recall)/(precision+recall+1e-9); f1_scores_aligned = f1_scores[:-1] 
                if len(f1_scores_aligned) > 0 and len(thresholds_pr) == len(f1_scores_aligned):
                    max_f1_idx = np.argmax(f1_scores_aligned); optimal_threshold_f1 = thresholds_pr[max_f1_idx]; max_f1_score = f1_scores_aligned[max_f1_idx]
                    precision_at_max_f1 = precision[max_f1_idx]; recall_at_max_f1 = recall[max_f1_idx]
                    results['Max_F1'] = round(max_f1_score, 4); results['Optimal_Threshold'] = round(optimal_threshold_f1, 4)
                    results['Precision_at_Max_F1'] = round(precision_at_max_f1, 4); results['Recall_at_Max_F1'] = round(recall_at_max_f1, 4)
                    print(f"Optimal Threshold (Max F1): {optimal_threshold_f1:.4f} -> F1={max_f1_score:.4f} (P={precision_at_max_f1:.4f}, R={recall_at_max_f1:.4f})")
                    if enable_plotting: # Plot F1 vs Threshold...
                         plt.figure(figsize=(10, 6)); plt.plot(thresholds_pr, f1_scores_aligned, label='F1', color='purple'); plt.plot(optimal_threshold_f1, max_f1_score, 'ro', ms=8, label=f'Max F1 @ T={optimal_threshold_f1:.3f}'); plt.xlabel('Threshold'); plt.ylabel('F1 Score'); plt.title('F1 vs Threshold (Stage 1)'); plt.legend(); plt.grid(True); plt.show()
                else: print("Could not calc F1 reliably.")
            except Exception as e: print(f"Error during F1 optimization: {e}")

            # --- Bayesian Filter Analysis ---
            bayes_precision, bayes_recall, bayes_f1 = -1.0, -1.0, -1.0 
            if X_test is not None and y_test is not None and machine_ids_test is not None:
                 print("\nApplying Bayesian filter (Stage 1 Eval)...")
                 try: # Bayes logic... 
                     test_results_df_s1 = pd.DataFrame({'Machine_ID': machine_ids_test, 'xgb_proba': y_pred_proba_test, 'Failure_Within_H': y_test.values}, index=X_test.index) 
                     test_results_with_bayes_s1 = apply_bayesian_filter(test_results_df_s1); # Use defaults
                     if not test_results_with_bayes_s1.empty:
                          y_true_bayes_s1 = test_results_with_bayes_s1['Failure_Within_H']; y_pred_bayes_s1 = test_results_with_bayes_s1['bayesian_alarm']
                          bayes_precision, bayes_recall, bayes_f1, _ = calculate_metrics_at_threshold(y_true_bayes_s1, y_pred_bayes_s1, 0.5) 
                          results['Bayes_F1'] = round(bayes_f1, 4); results['Bayes_Precision'] = round(bayes_precision, 4); results['Bayes_Recall'] = round(bayes_recall, 4)
                          print(f"Bayesian Metrics (S1): P={bayes_precision:.3f}, R={bayes_recall:.3f}, F1={bayes_f1:.3f}")
                          if enable_plotting: # Plot Bayes CM ...
                               cm_bayes_s1 = confusion_matrix(y_true_bayes_s1, y_pred_bayes_s1, labels=[0,1]); labels=[0,1]; cm_d_s1 = ConfusionMatrixDisplay(cm_bayes_s1, display_labels=[f'Actual {l}' for l in labels]); 
                               if cm_bayes_s1.size > 0 : fig_b, ax_b = plt.subplots(); cm_d_s1.plot(ax=ax_b, cmap='Oranges'); ax_b.set_title("CM (Bayesian - S1)"); plt.show()
                 except Exception as e: print(f"Error during Bayesian analysis: {e}")
            else: print("Skipping Bayesian filter: Missing data.")
                     
            # --- Save Stage 1 Predictions ---
            if y_pred_proba_test is not None and machine_ids_test is not None and y_test is not None:
                 print(f"\n--- Saving Stage 1 Predictions to {stage1_pred_output_path} ---")
                 try: # Save S1 preds logic... 
                      stage1_output = pd.DataFrame({'Machine_ID': machine_ids_test, 'xgb_proba': y_pred_proba_test, 'Failure_Within_H': y_test.values}, index=X_test.index) 
                      ensure_dir(stage1_pred_output_path); stage1_output.to_csv(stage1_pred_output_path)
                      print("S1 predictions saved.")
                 except Exception as e: print(f"Error saving S1 predictions: {e}")
            else: print("Skipping S1 predictions save.")
        else: print("\nModel not trained / Eval failed.")

    except Exception as pipeline_error: 
        print(f"Pipeline failed: {pipeline_error}")
        import traceback
        traceback.print_exc() 
        
    # --- Write results to CSV ---
    # This runs regardless, saving default values or last calculated ones
    save_results_to_csv(results_filepath, results) 
    
    print(f"\n--- Finished Run: W={feature_window_hrs}h, H={prediction_horizon_hrs}h ---")
    return results 

# --- Main Execution Guard ---
if __name__ == "__main__":
    # ... (Argument parsing - NO CHANGE) ...
    parser = argparse.ArgumentParser(description='Run Predictive Maintenance Model Training & Evaluation (Stage 1).')
    parser.add_argument('-w', '--window', type=int, default=DEFAULT_FEATURE_WINDOW_HOURS, help=f'Feature window size (W1) in hours (default: {DEFAULT_FEATURE_WINDOW_HOURS})')
    parser.add_argument('-H', '--horizon', type=int, default=DEFAULT_PREDICTION_HORIZON_HOURS, help=f'Prediction horizon (H1) in hours (default: {DEFAULT_PREDICTION_HORIZON_HOURS})')
    parser.add_argument('--plot', action='store_true', help='Enable plotting.')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model and feature list.')
    parser.add_argument('--s1_pred_output', type=str, default=STAGE1_PREDICTIONS_OUTPUT_FILE, help=f'Output file for stage 1 test predictions (default: {STAGE1_PREDICTIONS_OUTPUT_FILE})') 
    args = parser.parse_args()
    model_path = DEFAULT_MODEL_SAVE_PATH if args.save_model else None
    feature_cols_path = DEFAULT_FEATURE_COLS_SAVE_PATH if args.save_model else None
    if args.save_model: print(f"Model artifacts will be saved:\n  Model: {model_path}\n  Features: {feature_cols_path}")

    # --- Run the pipeline ---
    pipeline_results = {} # Initialize results dict here too
    try:
        # Call run_pipeline 
        pipeline_results = run_pipeline(
            feature_window_hrs=args.window, prediction_horizon_hrs=args.horizon, 
            test_size=DEFAULT_TEST_SET_SIZE, enable_plotting=args.plot,
            results_filepath=RESULTS_FILE, 
            model_save_path=model_path, 
            feature_cols_save_path=feature_cols_path, # Pass correctly
            stage1_pred_output_path=args.s1_pred_output )
            
        # --- >>> NEW: Save results HERE, immediately after successful run_pipeline <<< ---
        if pipeline_results: # Check if run_pipeline returned results
            # Ensure the save function is defined globally
            if 'save_results_to_csv' in globals():
                pass
                #save_results_to_csv(RESULTS_FILE, pipeline_results) 
            else: 
                print("Save function missing, cannot save results.")
        else: # Handle case where run_pipeline might return None on failure
            print("Pipeline did not return results.")
            # Optionally save default error results here if needed
            # if 'save_results_to_csv' in globals(): save_results_to_csv(RESULTS_FILE, {'Timestamp_Run': datetime.now()... etc.})


    except SystemExit: print("Pipeline exited.") 
    except Exception as e: 
        print(f"Critical error during pipeline execution: {e}")
        # Try saving default error results even on critical error
        if not pipeline_results: pipeline_results = {'Timestamp_Run': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'W_Hours': args.window, 'H_Hours': args.horizon, 'ROC_AUC': -1.0, 'PR_AUC': -1.0, 'Max_F1': 0.0, 'Optimal_Threshold': 0.5,'Precision_at_Max_F1': -1.0, 'Recall_at_Max_F1': -1.0,'Bayes_F1': -1.0, 'Bayes_Precision': -1.0, 'Bayes_Recall': -1.0 }
        if 'save_results_to_csv' in globals(): save_results_to_csv(RESULTS_FILE, pipeline_results) 
        else: print("Could not save error results.")
    # --- REMOVED finally block with save_results_to_csv ---
    
    print("\n--- predictive_model.py Script Finished ---")

'''
# --- Function to Save Grid Search Results --- 
def save_results_to_csv(filepath, results_data):
    """Appends results dictionary to a CSV file."""
    # ...(Same code as before)...
    print(f"Writing results to {filepath}...")
    try:
        ensure_dir(filepath) 
        file_exists = os.path.isfile(filepath); fieldnames = list(results_data.keys()) 
        with open(filepath, 'a', newline='', encoding='utf-8') as csvfile: 
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(filepath) == 0: writer.writeheader()  
            writer.writerow(results_data)
        print("Results written successfully.")
    except Exception as e: print(f"Error writing results to CSV: {e}")
'''