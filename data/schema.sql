-- Create machines table
CREATE TABLE machines (
    machine_id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100),
    type VARCHAR(50),
    location VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    installation_date TIMESTAMP,
    last_maintenance_date TIMESTAMP,
    specifications JSONB
);

-- Create sensor_data table
CREATE TABLE sensor_data (
    id SERIAL PRIMARY KEY,
    machine_id VARCHAR(10) REFERENCES machines(machine_id),
    timestamp TIMESTAMP NOT NULL,
    afr FLOAT,
    current FLOAT,
    pressure FLOAT,
    rpm FLOAT,
    temperature FLOAT,
    vibration FLOAT,
    CONSTRAINT unique_machine_reading UNIQUE(machine_id, timestamp)
);

-- Create predictions table
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    machine_id VARCHAR(10) REFERENCES machines(machine_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    predicted_failure_date TIMESTAMP,
    confidence FLOAT,
    model_version VARCHAR(20),
    prediction_details JSONB
);

-- Create simulations table
CREATE TABLE simulations (
    id SERIAL PRIMARY KEY,
    machine_id VARCHAR(10) REFERENCES machines(machine_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    scenario_type VARCHAR(50),
    parameters JSONB,
    results JSONB,
    created_by VARCHAR(50)
);

-- Create defaults table
CREATE TABLE defaults (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    key VARCHAR(50) NOT NULL,
    value TEXT,
    description TEXT,
    CONSTRAINT unique_category_key UNIQUE(category, key)
);

-- Create indexes for better query performance
CREATE INDEX idx_sensor_data_machine_id ON sensor_data(machine_id);
CREATE INDEX idx_sensor_data_timestamp ON sensor_data(timestamp);
CREATE INDEX idx_predictions_machine_id ON predictions(machine_id);
CREATE INDEX idx_simulations_machine_id ON simulations(machine_id);

-- Populate defaults table with sensor thresholds

-- AFR (Air-Fuel Ratio) defaults
INSERT INTO defaults (category, key, value, description)
VALUES 
    ('sensor_thresholds', 'afr_min', '10.0', 'Minimum acceptable Air-Fuel Ratio'),
    ('sensor_thresholds', 'afr_max', '15.0', 'Maximum acceptable Air-Fuel Ratio'),
    ('sensor_metadata', 'afr_unit', 'ratio', 'Measurement unit for Air-Fuel Ratio');

-- Current defaults
INSERT INTO defaults (category, key, value, description)
VALUES 
    ('sensor_thresholds', 'current_min', '20.0', 'Minimum acceptable Current'),
    ('sensor_thresholds', 'current_max', '40.0', 'Maximum acceptable Current'),
    ('sensor_metadata', 'current_unit', 'A', 'Measurement unit for Current (Ampere)');

-- Pressure defaults
INSERT INTO defaults (category, key, value, description)
VALUES 
    ('sensor_thresholds', 'pressure_min', '3.0', 'Minimum acceptable Pressure'),
    ('sensor_thresholds', 'pressure_max', '7.0', 'Maximum acceptable Pressure'),
    ('sensor_metadata', 'pressure_unit', 'bar', 'Measurement unit for Pressure');

-- RPM defaults
INSERT INTO defaults (category, key, value, description)
VALUES 
    ('sensor_thresholds', 'rpm_min', '2800', 'Minimum acceptable RPM'),
    ('sensor_thresholds', 'rpm_max', '3600', 'Maximum acceptable RPM'),
    ('sensor_metadata', 'rpm_unit', 'rpm', 'Measurement unit for RPM (Revolutions Per Minute)');

-- Temperature defaults
INSERT INTO defaults (category, key, value, description)
VALUES 
    ('sensor_thresholds', 'temperature_min', '65.0', 'Minimum acceptable Temperature'),
    ('sensor_thresholds', 'temperature_max', '95.0', 'Maximum acceptable Temperature'),
    ('sensor_metadata', 'temperature_unit', '°C', 'Measurement unit for Temperature (Degrees Celsius)');

-- Vibration defaults
INSERT INTO defaults (category, key, value, description)
VALUES 
    ('sensor_thresholds', 'vibration_min', '0.0', 'Minimum acceptable Vibration'),
    ('sensor_thresholds', 'vibration_max', '8.0', 'Maximum acceptable Vibration'),
    ('sensor_metadata', 'vibration_unit', 'mm/s', 'Measurement unit for Vibration');

-- System defaults
INSERT INTO defaults (category, key, value, description)
VALUES 
    ('system', 'data_retention_days', '365', 'Number of days to retain sensor data'),
    ('system', 'prediction_horizon_days', '30', 'Number of days in the future to make predictions'),
    ('system', 'maintenance_reminder_days', '7', 'Days before predicted failure to send maintenance reminder'),
    ('system', 'critical_alert_threshold', '0.8', 'Confidence threshold for critical alerts (0-1)'),
    ('system', 'warning_alert_threshold', '0.6', 'Confidence threshold for warning alerts (0-1)');

-- ML model defaults
INSERT INTO defaults (category, key, value, description)
VALUES 
    ('ml_params', 'training_window_days', '90', 'Days of data to use for model training'),
    ('ml_params', 'retraining_frequency_days', '30', 'Days between model retraining'),
    ('ml_params', 'feature_importance_threshold', '0.05', 'Minimum feature importance threshold for inclusion'),
    ('ml_params', 'default_model_type', 'random_forest', 'Default machine learning model type');