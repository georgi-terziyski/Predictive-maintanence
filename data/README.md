# PostgreSQL Database for Predictive Maintenance

This README provides instructions for setting up and using the PostgreSQL database for the predictive maintenance system.

## Database Schema

The database consists of the following tables:

1. **machines** - Information about each machine being monitored
2. **sensor_data** - Time-series sensor readings from the machines
3. **predictions** - Machine learning predictions for potential failures
4. **simulations** - Data about simulation scenarios run on machines
5. **maintenance** - Records of maintenance activities
6. **defaults** - System-wide default values and configuration

## Setup Instructions

### 1. Install PostgreSQL

If PostgreSQL is not already installed:

```bash
# On Windows:
# Download and install from https://www.postgresql.org/download/windows/

# On Ubuntu/Debian:
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
```

### 2. Create a PostgreSQL User

```bash
# Connect as the postgres superuser
sudo -u postgres psql

# In the PostgreSQL prompt, create a new user
CREATE USER your_username WITH PASSWORD 'your_password';

# Give the user permission to create databases
ALTER USER your_username CREATEDB;

# Exit PostgreSQL prompt
\q
```

### 3. Create the Database

```bash
# Create the database
createdb -U your_username predictive_maintenance
```

Or connect as the postgres user and create the database:

```bash
sudo -u postgres createdb predictive_maintenance
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE predictive_maintenance TO your_username;"
```

### 4. Initialize the Database Schema

```bash
# Run the schema creation script
psql -U your_username -d predictive_maintenance -f schema.sql

# Add the maintenance table
psql -U your_username -d predictive_maintenance -f maintenance_table.sql
```

### 5. Configure Environment Variables

Create a `.env` file in the project root directory:

```
DB_HOST=localhost
DB_NAME=predictive_maintenance
DB_USER=your_username
DB_PASSWORD=your_password
DATA_AGENT_PORT=5001
SUPERVISOR_PORT=5000
PREDICTION_AGENT_PORT=5002
SIMULATION_AGENT_PORT=5003
```

## Import Sensor Data

Use the provided Python script to import data from CSV files:

```bash
# Install required dependencies
pip install pandas psycopg2-binary python-dotenv

# Run the import script
python import_data.py
```

The script will:
1. Extract machine information from the CSV file
2. Import sensor readings into the database
3. Generate sample prediction and maintenance records (optional)

## API Endpoints

The Data Agent provides the following API endpoints:

### Health Check
- **GET /health** - Check the health status of the data agent and database connection

### Machines
- **GET /machines** - Get information about all machines

### Sensor Data
- **GET /predict?machine_id=M001** - Get sensor data for the last 4 days for a specific machine
- **GET /live_data** - Get the latest sensor readings for all machines
- **GET /live_data?machine_id=M001** - Get the latest sensor reading for a specific machine
- **GET /historical_data** - Get historical sensor data with optional filtering
  - Parameters: machine_id, limit, start_date, end_date

### Predictions
- **GET /predictions** - Get all prediction records
- **GET /predictions?machine_id=M001** - Get predictions for a specific machine
- **POST /predictions** - Create a new prediction record

### Maintenance
- **GET /maintenance** - Get all maintenance records
- **GET /maintenance?machine_id=M001&type=preventive&status=completed** - Get filtered maintenance records
- **POST /maintenance** - Create a new maintenance record

### System Defaults
- **GET /defaults** - Get all system default values
- **GET /defaults?category=sensor_thresholds** - Get defaults for a specific category

## Example Usage

### Get Machine Information

```bash
curl http://localhost:5001/machines
```

### Get Prediction Data for a Machine

```bash
curl http://localhost:5001/predict?machine_id=M001
```

### Add a Maintenance Record

```bash
curl -X POST http://localhost:5001/maintenance \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "M001",
    "maintenance_date": "2025-04-02T10:00:00",
    "completion_date": "2025-04-02T14:30:00",
    "maintenance_type": "corrective",
    "reason": "High vibration detected",
    "work_performed": "Replaced worn bearing",
    "technician_name": "Jane Smith",
    "technician_comments": "Bearing showed signs of excessive wear",
    "parts_replaced": "Main bearing assembly",
    "status": "completed",
    "downtime_hours": 4.5,
    "cost": 780.50
  }'
```

## Database Visualization

You can visualize the database using tools like:

1. **pgAdmin 4** - Official PostgreSQL administration platform
2. **DBeaver** - Universal database tool (free, open-source)
3. **TablePlus** - Modern, native database client (free + paid options)
