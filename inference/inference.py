import argparse
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

# File predictive_model.py must be included
from predictive_model import create_features

# --- Configurable Parameters ---
S1_MODEL_PATH    = 'inference/predictive_model_xgb_s1.joblib'
S1_FEATURES_PATH = 'inference/feature_columns_xgb_s1.joblib'
S2_MODEL_PATH    = 'inference/stage2_model_W84_H84_temp.joblib'
S2_FEATURES_PATH = 'inference/stage2_features_W84_H84_temp.joblib'
S2_ENCODER_PATH  = 'inference/stage2_class_encoder.joblib'

# --- JSON Sanitization ---
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

# --- Load model + feature list ---
def load_artifacts():
    s1_model      = joblib.load(S1_MODEL_PATH)
    s1_features   = joblib.load(S1_FEATURES_PATH)
    s2_model      = joblib.load(S2_MODEL_PATH)
    s2_features   = joblib.load(S2_FEATURES_PATH)
    label_encoder = joblib.load(S2_ENCODER_PATH)
    #labels = label_encoder.inverse_transform(s2_model.classes_)
    return s1_model, s1_features, s2_model, s2_features, label_encoder

# --- Load all input data ---
def load_raw_data(sensor_file, equipment_file, failure_file, maint_file):
    sensor    = pd.read_csv(sensor_file,  parse_dates  = ['Timestamp'])
    equipment = pd.read_csv(equipment_file)
    failures  = pd.read_csv(failure_file, parse_dates = ['Timestamp']) if os.path.exists(failure_file) else pd.DataFrame(columns=['Machine_ID', 'Timestamp', 'Failure_Type'])
    maint     = pd.read_csv(maint_file,   parse_dates = ['Timestamp']) if os.path.exists(maint_file)   else pd.DataFrame(columns=['Machine_ID', 'Timestamp', 'Maintenance_Action'])
    return sensor, equipment, failures, maint

# --- Prepare features ---
def prepare_features(df_sensor, df_equipment, hours_back, target_ts, window_size):
    time_cutoff = target_ts - timedelta(hours=hours_back)
    df_hist     = df_sensor[df_sensor['Timestamp'] <= target_ts]
    df_hist     = df_hist[df_hist['Timestamp'] >= time_cutoff]
    merged      = pd.merge(df_hist, df_equipment, on='Machine_ID', how='left')
    merged['Timestamp'] = pd.to_datetime(merged['Timestamp'])
    merged      = merged.set_index('Timestamp').sort_index()
    feats       = create_features(merged.copy(), window_size_hrs=window_size)
    feats       = feats[feats.index <= target_ts]
    return feats

# --- Apply XGB model safely ---
def apply_model(model, features_df, expected_features):
    for col in expected_features:
        if col not in features_df.columns:
            features_df[col] = 0.0
    features_df = features_df[expected_features]
    return model.predict_proba(features_df)[-1, 1]

def load_previous_result(machine_id, current_ts):
    prev_ts = current_ts - timedelta(minutes=30)
    fname = f"inference/preds/{machine_id}_{prev_ts:%Y%m%d_%H%M}.json"
    if os.path.exists(fname):
        with open(fname, "r") as f:
            return json.load(f)[0]  # in one-element list
    return None

def save_current_result(machine_id, current_ts, result):
    os.makedirs("inference/preds", exist_ok=True)
    fname = f"inference/preds/{machine_id}_{current_ts:%Y%m%d_%H%M}.json"
    with open(fname, "w") as f:
        json.dump([sanitize_for_json(result)], f, indent=2)

# --- Main inference logic ---
def inference(args):
    print("Loading artifacts...")
    s1_model, s1_feats, s2_model, s2_feats, label_encoder = load_artifacts()

    print("Loading data...")
    df_sensor, df_equipment, df_failure, df_maint = load_raw_data(
        args.input_data,
        args.equipment_file,
        args.failure_file,
        args.maintenance_file
    )

    infer_ts = pd.Timestamp(args.current_timestamp)
    machine_id = args.machine_id
    lookback_hours = max(args.window_s1, args.window_s2) #* 2
    #lookback_hours = (args.horizon_s2 - args.horizon_s1) + args.window_s2

    df_sensor = df_sensor[df_sensor['Machine_ID'] == machine_id]

    print("Generating S1 features...")
    feats_s1 = prepare_features(df_sensor, df_equipment, lookback_hours, infer_ts, args.window_s1)
    print("Generating S2 features...")
    feats_s2 = prepare_features(df_sensor, df_equipment, lookback_hours, infer_ts, args.window_s2)

    latest_s1 = feats_s1[feats_s1.index == feats_s1.index.max()]
    latest_s2 = feats_s2[feats_s2.index == feats_s2.index.max()]

    prob_s1 = apply_model(s1_model, latest_s1, s1_feats)

    result = {
        "timestamp": str(infer_ts),
        "machineID": machine_id,
        "status": "alert" if prob_s1 >= 0.5 else "ok",
        "stage1_probability": round(prob_s1, 4),
        "failure_predictions": []
    }

    if prob_s1 >= 0.5:
        prob_vec = s2_model.predict_proba(latest_s2[s2_feats])[-1]
        labels   = label_encoder.inverse_transform(s2_model.classes_)

        # Load previous prediction for Bayesian update
        prev_result = load_previous_result(machine_id, infer_ts)

        #"""
        for cls, p in zip(labels, prob_vec):
            hours_to_failure = None
            prior_p = None
            confidence_change = None

            if prev_result:
                prev_preds = {d["failure_type"]: d for d in prev_result.get("failure_predictions", [])}
                if cls in prev_preds:
                    prior_p = prev_preds[cls]["detection"]["probability"]
                    updated_p = (prior_p + p) / 2
                    confidence_change = round(updated_p - prior_p, 4)
                    p = updated_p

            result['failure_predictions'].append({
                "failure_type": cls,
                "detection": {
                    "detected": p >= 0.2,
                    "probability": round(p, 4),
                    "confidence": "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"
                },
                "bayesian_update": {
                    "hours_to_failure": hours_to_failure,
                    "prior_prediction": round(prior_p, 4) if prior_p is not None else None,
                    "confidence_change": confidence_change
                },
                "action_recommendation": {
                    "action": "monitor_closely" if p >= 0.3 else "monitor",
                    "urgency": "medium" if p >= 0.3 else "low",
                    "recommended_before": str(infer_ts + timedelta(hours=args.horizon_s1)) if p >= 0.3 else None
                }
            })
        #"""

        """
        cls_max, p_max = max(zip(labels, prob_vec), key=lambda x: x[1])

        result['failure_predictions'].append({
            "failure_type": cls_max,
            "detection": {
                "detected": p_max >= 0.2,
                "probability": round(p_max, 4),
                "confidence": "high" if p_max >= 0.6 else "medium" if p_max >= 0.3 else "low"
            },
            "bayesian_update": {
                "hours_to_failure": None,
                "prior_prediction": None,
                "confidence_change": None
            },
            "action_recommendation": {
                "action": "monitor_closely" if p_max >= 0.3 else "monitor",
                "urgency": "medium" if p_max >= 0.3 else "low",
                "recommended_before": str(infer_ts + timedelta(hours=args.horizon_s1)) if p_max >= 0.3 else None
            }
        })
        """

        

    print("\n--- Inference Results (JSON) ---")
    print(json.dumps([sanitize_for_json(result)], indent=2))
    save_current_result(machine_id, infer_ts, result)
    #print(s2_model.classes_)
    #print(labels)
    print("\n--- Inference Script Finished ---")
    
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run dual-stage inference for a given machine and timestamp.")
    parser.add_argument('--current_timestamp', type=str, required=True, help='Timestamp to run inference for')
    parser.add_argument('--input_data', type=str, required=True, help='Sensor CSV file')
    parser.add_argument('--equipment_file', type=str, default='inference/data/10apr/equipment_usage.csv')
    parser.add_argument('--failure_file', type=str, default='inference/data/10apr/failure_logs.csv')
    parser.add_argument('--maintenance_file', type=str, default='inference/data/10apr/maintenance_history.csv')
    parser.add_argument('--machine_id', type=str, required=True, help='Machine ID to run inference for')
    parser.add_argument('--window_s1', type=int, default=24, help='Feature window for Stage 1')
    parser.add_argument('--horizon_s1', type=int, default=48, help='Prediction horizon for Stage 1')
    parser.add_argument('--window_s2', type=int, default=84, help='Feature window for Stage 2')
    parser.add_argument('--horizon_s2', type=int, default=84, help='Prediction horizon for Stage 2 (not used yet)')

    args = parser.parse_args()
    infer_ts = pd.Timestamp(args.current_timestamp)
    result = inference(args)
    # Result is already printed and saved inside the inference function
