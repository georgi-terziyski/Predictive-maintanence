import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
# Ensure scipy is installed: pip install scipy
from scipy.stats import gamma as gamma_dist 

# --- Configuration Constants ---
START_DATE  = '1922-01-01'
END_DATE    = '2023-12-31'
SENSOR_FREQ = '1800s' # 30 minutes
MACHINE_IDS = [f'M{i:03d}' for i in range(1, 11)]  # 10 machines

# --- Failure Generation Configuration ---
FAILURE_POTENTIAL_WEEK_INTERVAL = 3       # Potential arises every 2 weeks
MAX_LEAD_TIME_HOURS             = 7 * 24  # Max 1 week from potential to failure (168 hours)
MIN_LEAD_TIME_HOURS             = 5.0     # Minimum required lead time

# --- Gamma Distribution Parameters for Lead Time ---
# Aim for a peak around 60 hours (adjust k and theta)
TARGET_PEAK_HOURS = 60 
GAMMA_SHAPE       = 5.0  # k (Shape parameter)
# Calculate scale (theta) for the desired peak: peak = (k-1)*theta
GAMMA_SCALE       = TARGET_PEAK_HOURS / (GAMMA_SHAPE - 1) if GAMMA_SHAPE > 1 else 15.0 

# --- Trend Configuration (Revised with SUBTLER max_delta) ---
# Goal: Trends should be statistically detectable in features (mean, std, etc.) 
# over a window, but not visually obvious in raw plots. Delta values
# are now smaller relative to base operational ranges and thresholds.
TREND_CONFIG = {
    'Bearing Failure': {
        'sensors': {'Vibration': 2.0, 'Current': 2.5, 'Temperature': 4.0}, # Slightly increased
        'trend_type': 'quadratic', 
        'start_ratio': 0.1 
    },
    'Motor Burnout': {
        'sensors': {'Current': 6.5, 'Temperature': 9.0, 'Vibration': 0.7}, # Slightly increased
        'trend_type': 'linear', 
        'start_ratio': 0.4 
    },
    'Overheating': {
        'sensors': {'Temperature': 10.0, 'Pressure': 1.0, 'Current': 1.5}, # Slightly increased
        'trend_type': 'linear',
        'start_ratio': 0.2
    },
    'Pressure System Failure': {
        'sensors': {'Pressure': 1.5, 'Vibration': 0.5}, # Slightly increased
        'trend_type': 'linear',
        'start_ratio': 0.2 # Adjusted start ratio slightly
    },
    'Carbon Buildup': {
        'sensors': {'AFR': -1.5, 'Temperature': 5.0, 'RPM_std_factor': 1.4}, # Slightly increased effect
        'trend_type': 'linear',
        'start_ratio': 0.05 
    },
    'Electrical Malfunction': {
        'sensors': {'Current': 5.0, 'Vibration': 1.5}, # Slightly increased spike delta
        'trend_type': 'spikes', 
        'start_ratio': 0.6,
        'spike_probability': 0.18 # Slightly more frequent spikes
    }
}

# --- Functions ---

def generate_equipment_usage():
    """Generates static equipment usage data."""
    print("Generating Equipment Usage data...")
    data = [[machine, random.randint(1, 10), random.randint(50000, 200000)] for machine in MACHINE_IDS]
    print("Equipment Usage data generation completed.")
    return pd.DataFrame(data, columns=['Machine_ID', 'Equipment_Age (Years)', 'Usage_Cycles'])

def plan_potential_failures(start_date_str, end_date_str, machine_ids):
    """Plans potential failure events with controlled timing."""
    print("Planning potential failures...")
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    planned_failures = []
    
    failure_types = list(TREND_CONFIG.keys())

    print(f"Gamma distribution parameters: Shape(k)={GAMMA_SHAPE:.2f}, Scale(theta)={GAMMA_SCALE:.2f}")
    print(f"Targeting peak around {TARGET_PEAK_HOURS}h, with minimum lead time {MIN_LEAD_TIME_HOURS}h.")

    current_date = start_date
    week_counter = 0
    last_potential_week = {machine: -FAILURE_POTENTIAL_WEEK_INTERVAL for machine in machine_ids} 

    while current_date < end_date:
        is_potential_week = (week_counter % FAILURE_POTENTIAL_WEEK_INTERVAL == 0)
        week_start = current_date
        
        if is_potential_week:
            for machine in machine_ids:
                # Generate potential event time within the week
                potential_offset_seconds = random.uniform(0, 7 * 24 * 3600 - 1)
                potential_start_time = week_start + timedelta(seconds=potential_offset_seconds)
                
                if potential_start_time >= end_date: continue

                # Generate lead time using Gamma distribution + min offset
                generated_gamma_h = -1
                loop_guard = 0 
                while (generated_gamma_h <= 0) and loop_guard < 100:
                    generated_gamma_h = gamma_dist.rvs(a=GAMMA_SHAPE, scale=GAMMA_SCALE) 
                    loop_guard += 1
                
                if generated_gamma_h <= 0: 
                    print("Warning: Gamma generation resulted in <=0, using mean.")
                    generated_gamma_h = GAMMA_SHAPE * GAMMA_SCALE

                # Add minimum lead time and cap at maximum
                lead_time_h = MIN_LEAD_TIME_HOURS + generated_gamma_h
                lead_time_h = min(lead_time_h, MAX_LEAD_TIME_HOURS)

                actual_failure_time = potential_start_time + timedelta(hours=lead_time_h)

                if actual_failure_time >= end_date: continue 

                failure_type = random.choice(failure_types)

                planned_failures.append({
                    'Machine_ID': machine,
                    'Potential_Start_Time': potential_start_time,
                    'Actual_Failure_Time': actual_failure_time,
                    'Lead_Time_Hours': lead_time_h, 
                    'Failure_Type': failure_type
                })
                last_potential_week[machine] = week_counter

        current_date += timedelta(days=7)
        week_counter += 1
        
    print(f"Planned {len(planned_failures)} potential failure events.")
    return planned_failures

def generate_base_sensor_data(start_date_str, end_date_str, freq, machine_ids):
    """Generates base sensor data within normal operating ranges."""
    print("Generating base sensor data...")
    date_range = pd.date_range(start=start_date_str, end=end_date_str, freq=freq)
    sensor_data = []

    total_records = len(date_range) * len(machine_ids)
    processed = 0
    start_time_gen = datetime.now()

    for machine in machine_ids:
        for timestamp in date_range:
            processed += 1
            if processed % 10000 == 0:
                 elapsed = (datetime.now() - start_time_gen).total_seconds()
                 rate = processed / elapsed if elapsed > 0 else 0
                 print(f"  Processed {processed}/{total_records} ({processed/total_records:.1%}) base records... Rate: {rate:.0f} rec/s", end='\r')

            # Normal operating ranges (relatively stable)
            afr = round(np.random.uniform(11.5, 14.5), 2) 
            current = round(np.random.uniform(22, 32), 2) 
            pressure = round(np.random.uniform(3.5, 5.0), 2) 
            rpm_base = random.randint(3000, 3400) 
            temperature = round(np.random.uniform(68, 82), 2) 
            vibration = round(np.random.uniform(1.5, 4.5), 2) 

            # Base RPM variance 
            rpm = rpm_base + random.randint(-50, 50) 

            sensor_data.append([timestamp, machine, afr, current, pressure, rpm, temperature, vibration])

    print(f"\nBase sensor data generation complete. Total Records: {len(sensor_data)}")
    base_df = pd.DataFrame(sensor_data, columns=['Timestamp', 'Machine_ID', 'AFR', 'Current', 'Pressure', 'RPM', 'Temperature', 'Vibration'])
    return base_df.set_index('Timestamp')


def inject_failure_trends(sensor_df, base_df_for_stats, planned_failures, trend_config):
    """Injects subtle failure trends into sensor data."""
    print("Injecting failure trends into sensor data...")
    sensor_df_with_trends = sensor_df.copy()
    
    total_failures_to_inject = len(planned_failures)
    injected_count = 0
    start_time_inj = datetime.now()

    # Pre-calculate base stats per machine for RPM_std_factor efficiency
    base_stats_rpm = base_df_for_stats.groupby('Machine_ID')['RPM'].agg(['mean', 'std']).fillna(0)
    base_stats_rpm['std'] = base_stats_rpm['std'].replace(0, 10) # Avoid std=0

    for failure_plan in planned_failures:
        injected_count += 1
        if injected_count % 10 == 0:
            elapsed = (datetime.now() - start_time_inj).total_seconds()
            rate = injected_count / elapsed if elapsed > 0 else 0
            print(f"  Injecting trend {injected_count}/{total_failures_to_inject}... Rate: {rate:.1f} trends/s", end='\r')

        machine = failure_plan['Machine_ID']
        start_trend_potential = failure_plan['Potential_Start_Time']
        fail_time = failure_plan['Actual_Failure_Time']
        failure_type = failure_plan['Failure_Type']
        lead_time_total_seconds = (fail_time - start_trend_potential).total_seconds()

        if lead_time_total_seconds <= 0: continue 

        config = trend_config.get(failure_type)
        if not config: continue 

        trend_indices = sensor_df_with_trends[
            (sensor_df_with_trends['Machine_ID'] == machine) &
            (sensor_df_with_trends.index >= start_trend_potential) &
            (sensor_df_with_trends.index < fail_time) 
        ].index

        if trend_indices.empty: continue 

        trend_start_offset = lead_time_total_seconds * config['start_ratio']
        effective_trend_start_time = start_trend_potential + timedelta(seconds=trend_start_offset)
        
        # Get base RPM stats for this machine
        machine_base_rpm_mean = base_stats_rpm.loc[machine, 'mean']
        machine_base_rpm_std = base_stats_rpm.loc[machine, 'std']

        for idx in trend_indices:
            time_elapsed_seconds = (idx - start_trend_potential).total_seconds()
            # progress: 0 to 1 over the *entire* lead time period
            progress = min(max(time_elapsed_seconds / lead_time_total_seconds, 0.0), 1.0) 

            if idx >= effective_trend_start_time:
                # effective_progress: 0 to 1 over the duration the trend is *active*
                time_since_trend_start = (idx - effective_trend_start_time).total_seconds()
                duration_of_this_trend = lead_time_total_seconds * (1.0 - config['start_ratio'])
                effective_progress = min(max(time_since_trend_start / duration_of_this_trend, 0.0), 1.0) if duration_of_this_trend > 0 else 1.0

                for sensor, max_delta in config['sensors'].items():
                    
                    # Special handling for RPM std factor
                    if sensor == 'RPM_std_factor':
                         std_increase_factor = 1 + (max_delta - 1) * effective_progress 
                         # Generate new RPM based on increased std around the machine's base mean
                         new_rpm = np.random.normal(loc=machine_base_rpm_mean, scale=machine_base_rpm_std * std_increase_factor) 
                         sensor_df_with_trends.loc[idx, 'RPM'] = round(new_rpm) 
                         continue # Move to next sensor

                    # Standard sensor handling
                    try:
                         current_value = sensor_df_with_trends.loc[idx, sensor]
                    except KeyError:
                         print(f"\nWarning: Sensor '{sensor}' defined in TREND_CONFIG for '{failure_type}' not found in DataFrame columns. Skipping.")
                         continue 
                         
                    delta = 0
                    
                    # Calculate delta based on trend type
                    if config['trend_type'] == 'linear':
                        delta = max_delta * effective_progress
                    elif config['trend_type'] == 'quadratic':
                        delta = max_delta * (effective_progress ** 2)
                    elif config['trend_type'] == 'exponential': 
                        # Cap exponential growth to avoid extreme values, scale to reach max_delta at end
                        exp_factor = min(effective_progress * 4, 4) # Limit effective exponent
                        delta = max_delta * (np.exp(exp_factor) - 1) / (np.exp(4) - 1) 
                    elif config['trend_type'] == 'spikes':
                         if random.random() < config.get('spike_probability', 0.1):
                             # Spike size can also vary
                             delta = max_delta * random.uniform(0.5, 1.0) 
                         else:
                             delta = 0 

                    new_value = current_value + delta
                    
                    # Add reduced noise - make trends smoother but not perfect
                    noise_std_dev = abs(max_delta * 0.05) # Noise std dev is 5% of max_delta
                    noise = np.random.normal(0, noise_std_dev) if noise_std_dev > 0 else 0
                    new_value += noise

                    # Apply and round where necessary
                    if sensor in ['AFR', 'Current', 'Pressure', 'Temperature', 'Vibration']:
                         sensor_df_with_trends.loc[idx, sensor] = round(new_value, 2)
                    else:
                         sensor_df_with_trends.loc[idx, sensor] = new_value 


    print(f"\nFailure trend injection complete. Processed {injected_count} plans.")
    return sensor_df_with_trends


def generate_failure_logs_from_plan(planned_failures):
    """Generates the failure_logs DataFrame from the plan."""
    print("Generating failure logs from plan...")
    if not planned_failures:
        return pd.DataFrame(columns=['Machine_ID', 'Timestamp', 'Failure_Type'])
        
    failure_data = [{
        'Machine_ID': p['Machine_ID'],
        'Timestamp': p['Actual_Failure_Time'],
        'Failure_Type': p['Failure_Type']
    } for p in planned_failures]
    
    df = pd.DataFrame(failure_data)
    df = df.sort_values(by=['Machine_ID', 'Timestamp']).reset_index(drop=True)
    print(f"Generated {len(df)} failure log entries.")
    return df

def generate_maintenance_logs(failure_df):
    """Generates maintenance logs following failures."""
    print("Generating maintenance logs...")
    maintenance_data = []
    maintenance_actions = { 
        'Bearing Failure': 'Bearing Replacement',
        'Motor Burnout': 'Motor Servicing',
        'Overheating': 'Cooling System Check',
        'Pressure System Failure': 'Pressure Adjustment',
        'Carbon Buildup': 'Carbon Cleanup',
        'Electrical Malfunction': 'Electrical Inspection'
    }

    for _, row in failure_df.iterrows():
        # Maintenance occurs after a random delay
        maintenance_time = row['Timestamp'] + timedelta(hours=random.randint(2, 8)) 
        action = maintenance_actions.get(row['Failure_Type'], 'General Inspection') 
        maintenance_data.append([row['Machine_ID'], maintenance_time, action])

    df = pd.DataFrame(maintenance_data, columns=['Machine_ID', 'Timestamp', 'Maintenance_Action'])
    print(f"Generated {len(df)} maintenance log entries.")
    return df

# --- MODIFIED: apply_downtime ---
def apply_downtime(sensor_df, failure_df, maintenance_df):
    """Sets sensor values to NaN during downtime (failure to maintenance)."""
    print("Applying downtime (setting sensor values to NaN during failure/maintenance)...")
    sensor_df_operational = sensor_df.copy()
    
    # Find corresponding maintenance time for each failure
    fail_maint = pd.merge(failure_df, maintenance_df, on='Machine_ID', suffixes=('_fail', '_maint'))
    fail_maint = fail_maint[fail_maint['Timestamp_maint'] > fail_maint['Timestamp_fail']]
    fail_maint = fail_maint.sort_values('Timestamp_maint').groupby(['Machine_ID', 'Timestamp_fail']).first().reset_index()

    nan_count = 0
    sensor_columns = ['AFR', 'Current', 'Pressure', 'RPM', 'Temperature', 'Vibration'] # Columns to set to NaN

    for _, row in fail_maint.iterrows():
        machine = row['Machine_ID']
        fail_time = row['Timestamp_fail']
        maint_time = row['Timestamp_maint']

        # Find indices during downtime (inclusive of fail_time, exclusive of maint_time)
        downtime_mask = (
            (sensor_df_operational['Machine_ID'] == machine) &
            (sensor_df_operational.index >= fail_time) &
            (sensor_df_operational.index < maint_time) 
        )
        
        if downtime_mask.any():
             # Set specified sensor columns to NaN for the identified indices
             sensor_df_operational.loc[downtime_mask, sensor_columns] = np.nan
             nan_count += downtime_mask.sum() * len(sensor_columns) # Count total NaN values set


    print(f"Set {nan_count} sensor readings to NaN due to downtime.")
    return sensor_df_operational


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Extended Data Generation (v5 - Subtle Trends, NaN Downtime) ---")
    
    # 1. Generate equipment usage info
    equipment_usage_df = generate_equipment_usage()

    # 2. Plan potential failures (includes timing)
    planned_failures_list = plan_potential_failures(START_DATE, END_DATE, MACHINE_IDS)

    # 3. Generate base sensor data (stable operation)
    base_sensor_df = generate_base_sensor_data(START_DATE, END_DATE, SENSOR_FREQ, MACHINE_IDS)

    # 4. Inject subtle failure trends
    sensor_df_with_trends = inject_failure_trends(base_sensor_df, base_sensor_df, planned_failures_list, TREND_CONFIG) # Pass base_df for stats

    # 5. Generate failure logs from the plan
    failure_logs_df = generate_failure_logs_from_plan(planned_failures_list)

    # 6. Generate maintenance logs
    maintenance_history_df = generate_maintenance_logs(failure_logs_df)

    # 7. Apply downtime (set values to NaN)
    final_sensor_df = apply_downtime(sensor_df_with_trends, failure_logs_df, maintenance_history_df)

    # 8. Save to CSV files
    print("\nSaving CSV Files...")
    # Save sensor data with NaNs
    final_sensor_df.reset_index().to_csv('data/07may/sensor_data_ext.csv', index=False) 
    equipment_usage_df.to_csv('data/07may/equipment_usage_ext.csv', index=False)
    failure_logs_df.to_csv('data/07may/failure_logs_ext.csv', index=False)
    maintenance_history_df.to_csv('data/07may/maintenance_history_ext.csv', index=False)

    print("\n--- Extended Data Generation Complete. CSV files saved (with '_ext' suffix). ---")