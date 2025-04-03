# Digital Twin for Predictive Maintenance

A multi-agent system for predictive maintenance of industrial machines using sensor data and machine learning.

## System Architecture

```mermaid
graph TD
    A[Supervisor] --> B[Data Agent]
    A --> C[Prediction Agent] 
    A --> D[Simulation Agent]
    A --> E[Analytics Agent]
    B --> F[(Database)]
    C --> G[ML Model]
    D --> H[Scenario Engine]
    E --> I[Analysis Tools]
```

## Components

1. **Supervisor Service** - Coordinates communication between agents (port 5000)
2. **Data Agent** - Fetches live machine data from database (port 5001)
3. **Prediction Agent** - Runs ML predictions on incoming data (port 5002)
4. **Simulation Agent** - Performs what-if scenario modeling (port 5003)  
5. **Analytics Agent** - Analyzes results and compares scenarios (port 5004)

## Pre-requisites

Python version 3.10.16

## Installation

1. Clone the repository
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Copy `.env.example` to `.env`
2. Update environment variables:
   - Database credentials
   - Model paths
   - API keys
   - Port configurations

## Running the System

Start each service in separate terminals:

```bash
# Supervisor
python digital_twin/supervisor/app.py

# Data Agent  
python digital_twin/data_agent/app.py

# Prediction Agent
python digital_twin/prediction_agent/app.py

# Simulation Agent
python digital_twin/simulation_agent/app.py

# Analytics Agent
python digital_twin/analytics_agent/app.py
```

## API Documentation

### Supervisor Endpoints

- `GET /health` - System health check
- `GET /machine_list` - Get simplified list of machine IDs and names
- `GET /machine_defaults` - Get machine-specific threshold values (averaged min/max)
- `POST /predict` - Get machine predictions
- `POST /simulate` - Run scenario simulation
- `POST /analyze` - Analyze results
- `POST /compare` - Compare scenarios

### Data Agent Endpoints

- `GET /health` - Agent health check
- `GET /machines` - Get all machine details
- `GET /machine_list` - Get simplified list of machine IDs and names
- `GET /live_data` - Get latest sensor readings
- `GET /historical_data` - Get historical sensor data
- `GET /defaults` - Get system and machine-specific thresholds
  - Now supports `machine_id` parameter to return machine-specific values
  - Returns averaged values for min/max pairs (except vibration)

## Example Usage

```python
import requests

# Get system health
response = requests.get("http://localhost:5000/health")
print(response.json())

# Get machine list
machines = requests.get("http://localhost:5000/machine_list")
print(machines.json())

# Get machine-specific thresholds (full details via data agent)
defaults = requests.get("http://localhost:5001/defaults?category=sensor_thresholds")
print(defaults.json())

# Get machine-specific thresholds with averaged values (via supervisor)
machine_defaults = requests.get("http://localhost:5000/machine_defaults?machine_id=M001")
print(machine_defaults.json())
# Example output:
# {
#   "machine_id": "M001",
#   "machine_name": "Precision milling machine", 
#   "machine_type": "Milling",
#   "afr": "12.5",
#   "rpm": "3200", 
#   "current": "30.0",
#   "pressure": "5.0",
#   "temperature": "72.5",
#   "vibration_max": "5.0"
# }

# Get prediction
prediction = requests.post("http://localhost:5000/predict", json={
    "machine_id": "M001",
    "sensor_readings": {...}
})
print(prediction.json())
