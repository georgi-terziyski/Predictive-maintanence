# Prediction Agent

The Prediction Agent is responsible for making failure predictions based on machine sensor data. It fetches historical sensor data from the Data Agent, processes this data through a machine learning model, and returns predictions about when failures might occur.

## Overview

This agent:
1. Retrieves 4 days of historical sensor data from the Data Agent
2. Processes this data to extract meaningful features
3. Feeds these features into a machine learning model
4. Returns predictions about potential machine failures
5. Stores prediction results in the database via Data Agent

## API Endpoints

### Health Check
```
GET /health
```
Returns the current status of the Prediction Agent, including model availability and Data Agent connectivity.

### Predict Failure
```
POST /predict
```
Makes a prediction for the specified machine.

**Request Body**:
```json
{
  "machine_id": "M001"
}
```

**Response Body**:
```json
{
  "machine_id": "M001",
  "predicted_failure_date": "2025-05-02T14:30:00.000Z",
  "days_to_failure": 30,
  "confidence": 0.85,
  "contributing_factors": ["high vibration", "temperature fluctuation"],
  "prediction_id": 123,
  "timestamp": "2025-04-02T14:30:00.000Z"
}
```

## Configuration

The Prediction Agent requires the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| DATA_AGENT_URL | URL of the Data Agent | http://localhost:5001 |
| PREDICTION_AGENT_PORT | Port for the Prediction Agent | 5002 |
| MODEL_PATH | Path to the ML model file | agents/models/failure_prediction_model.pkl |
| PREDICTION_HORIZON_DAYS | Number of days to predict into the future | 30 |

## Feature Engineering

The Prediction Agent extracts the following features from sensor data:

- Average readings (temperature, vibration, pressure, etc.)
- Standard deviations (to detect variability)
- Trends over time
- Maximum values
- Count of outlier readings

These features are then passed to the ML model to generate predictions.

## Machine Learning Model

The system uses a placeholder model that predicts days until failure. To use a real model:

1. Train your model using historical sensor data
2. Save it using joblib:
   ```python
   import joblib
   joblib.dump(model, 'agents/models/failure_prediction_model.pkl')
   ```
3. Update the MODEL_PATH in your .env file

## Integration with Data Agent

The Prediction Agent relies on the Data Agent for:
1. Fetching historical sensor data via the `/predict` endpoint
2. Storing prediction results via the `/predictions` endpoint

Make sure the Data Agent is running and properly configured before starting the Prediction Agent.
