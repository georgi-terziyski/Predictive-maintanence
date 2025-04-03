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

-- Create defaults table with machine_id reference
CREATE TABLE defaults (
    id SERIAL PRIMARY KEY,
    machine_id VARCHAR(10) REFERENCES machines(machine_id),
    category VARCHAR(50) NOT NULL,
    key VARCHAR(50) NOT NULL,
    value TEXT,
    description TEXT,
    CONSTRAINT unique_machine_category_key UNIQUE(machine_id, category, key)
);

-- Create maintenance table
CREATE TABLE maintenance (
    id SERIAL PRIMARY KEY,
    machine_id VARCHAR(10) REFERENCES machines(machine_id),
    maintenance_date TIMESTAMP NOT NULL,
    completion_date TIMESTAMP,
    maintenance_type VARCHAR(50) NOT NULL,
    reason TEXT NOT NULL,
    work_performed TEXT,
    technician_name VARCHAR(100),
    technician_comments TEXT,
    parts_replaced TEXT,
    status VARCHAR(20) DEFAULT 'completed',
    downtime_hours FLOAT,
    cost DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for better query performance
CREATE INDEX idx_maintenance_machine_id ON maintenance(machine_id);
CREATE INDEX idx_maintenance_date ON maintenance(maintenance_date);

-- Create function to update last maintenance date
CREATE OR REPLACE FUNCTION update_machine_last_maintenance()
RETURNS TRIGGER AS $$
BEGIN
    -- Update the machines table last_maintenance_date when maintenance is completed
    IF NEW.status = 'completed' THEN
        UPDATE machines
        SET last_maintenance_date = NEW.completion_date
        WHERE machine_id = NEW.machine_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to call the function after maintenance insert/update
CREATE TRIGGER after_maintenance_update
AFTER INSERT OR UPDATE ON maintenance
FOR EACH ROW
EXECUTE FUNCTION update_machine_last_maintenance();

-- Create index for better query performance
CREATE INDEX idx_defaults_machine_id ON defaults(machine_id);

-- Create indexes for better query performance
CREATE INDEX idx_sensor_data_machine_id ON sensor_data(machine_id);
CREATE INDEX idx_sensor_data_timestamp ON sensor_data(timestamp);
CREATE INDEX idx_predictions_machine_id ON predictions(machine_id);
CREATE INDEX idx_simulations_machine_id ON simulations(machine_id);
