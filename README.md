# TSM MachLeData - Flight Price Prediction

ML pipeline with drift detection, automated retraining, and model promotion to a FastAPI server.

## Dataset
Indian flight price prediction dataset obtained from “Ease My Trip” website.

[Kaggle - Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

*Note: Prices are expressed in Indian Rupees.*

## Run pipeline

```
python (or python3) -m "src.mlflow_pipeline"
```

**Before running above command, make sure local FastAPI Server is running (see section below)**

### Use cases
o test different scenarios, the following parameters can be modified before execution:
- DATA_DRIFT (False/True): Simulate artificial data drift for the current week
- CONCEPT_DRIFT = (False/True): Simulate artificial concept drift for the current week
- CURR_WEEK [7-13]: Fix the current week
- LAST_TRAIN_WEEK [6-12]: Fix the last training week (useful for testing expanding data drift)

### Notable scenarios
- Without any simulated drift, running with `CURR_WEEK = 9` will automatically trigger drift detection and force retraining.
Running a first instance with this week is useful to test data drift, concept drift, retraining and model promotion.
- Running on next weeks will not trigger drift detection. Therefore, it helps to test model drift.

Other combinations with or without simulated drift can also be tested.

## Setup docker - FastAPI Server

To run the FastAPI server locally:

```
# Start development container
docker compose -f docker-compose.dev.yml up

# Access API
http://localhost:52001/docs

# Stop cleanly
docker compose down
```

Note:
- docker-compose.prod.yml is used for deployment on GCloud. The image is pull directly from Github package.
- Because FastAPI uses SQLite during local runs, stored models are not persistent once the container restarts. Therefore, test scenarios should be performed in a single session.

## MLflow Usage

To launch the MLflow server and inspect pipeline logs:

```
mlflow server

# access Mflow
http://localhost:5000
```