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
