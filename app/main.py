from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib 
import numpy as np
import traceback
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        # 1. Initializing parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.cost_history = []

        # 2. Gradient Descent
        for i in range(self.n_iterations):
            # Calculate predictions
            y_pred = X @ self.weights + self.bias

            # Calculating cost (MSE)
            cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)

            # Calculating the gradients
            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Updating parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Printing cost periodically
            if (i+1) % (self.n_iterations // 10) == 0 or i == 0: 
                print(f"Iteration {i+1}/{self.n_iterations}, Cost: {cost:.4f}")

    def predict(self, X, scaled=True):
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet")
        return X @ self.weights + self.bias



print("Loading applications artifacts...")
ARTIFACTS = {}
STRUCTURE_TYPES = ["Residential", "Commercial", "Industrial", "Mixed-use"]
PREFIX_MAP = {
    "Residential": "res",
    "Commercial": "com",
    "Industrial": "ind",
    "Mixed-use": "mix"
}

for structure_type in STRUCTURE_TYPES:
    prefix = PREFIX_MAP[structure_type]
    try:
        model = joblib.load(f"app/models/{prefix}_model.joblib")
        scalers = joblib.load(f"app/models/{prefix}_scalers.joblib")

        ARTIFACTS[structure_type.lower()] = {'model': model, 'scalers': scalers}
        print(f"-- Successfully loaded artifacts for {structure_type}")
    except FileNotFoundError:
        print(f"Artifacts for {structure_type} not found. This type will be unavailable.")

print("Application loaded succesfully.")

app = FastAPI(
    title="Electricity Cost Prediction API",
    description="An API to predict electricity costs based on building features.",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    structure_type: str
    site_area: float
    water_consumption: float
    utilisation_rate: float
    resident_count: int
    class Config:
        json_schema_extra = {
            "example": {
                "structure_type": "Residential",
                "site_area": 3100.0,
                "water_consumption": 6200.0,
                "utilisation_rate": 90.0,
                "resident_count": 15
            }
        }

def make_prediction(model, scalers, X_raw):
    
    # 1. Scale the raw input features
    x_scaler_params = scalers['x_scaler_params']
    X_scaled = (X_raw - x_scaler_params['min']) / (x_scaler_params['max'] - x_scaler_params['min'])

    # 2. Make prediction on scaled data
    prediction_scaled = model.predict(X_scaled)

    # 3. Inverse-scale the prediction to get the final result
    y_scaler_params = scalers['y_scaler_params']
    prediction_raw = prediction_scaled * (y_scaler_params['max'] - y_scaler_params['min']) + y_scaler_params['min']

    return prediction_raw.flatten()[0]

@app.post("/predict")
def predict_electricity_cost(data: PredictionRequest):
    structure = data.structure_type.lower()

    if structure not in ARTIFACTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Structure Type. Available types: {list(ARTIFACTS.keys())}"
        )
    
    model_artifacts = ARTIFACTS[structure]
    model = model_artifacts['model']
    scalers = model_artifacts['scalers']
    x_scaler_params = scalers['x_scaler_params']

    try:
        # 1. Preparing the feature array based on the structure type
        if structure in ['commercial', 'industrial']:
            features = [data.site_area, data.water_consumption, data.utilisation_rate]
        else:
            features = [data.site_area, data.water_consumption, data.utilisation_rate, data.resident_count]

        X_raw = np.array([features], dtype=np.float64)

        # Comparing the number of input features with the number of expected features.
        num_input_features = X_raw.shape[1]
        num_scaler_features = len(x_scaler_params['min'])

        if num_input_features != num_scaler_features:
            raise ValueError(
                f"Shape Mismatch: The model for '{data.structure_type}' expects {num_scaler_features} features, "
                f"but {num_input_features} were provided."
            )
        prediction = make_prediction(model, scalers, X_raw)
        
        return {"predicted_cost": round(prediction, 2)}

    # This will catch any error during prediction
    # and return a helpful, detailed error message instead of a generic 500.
    except Exception as e:
        print("--- An Error Occurred During Prediction ---")
        traceback.print_exc() 
        print("-----------------------------------------")
        
        raise HTTPException(
            status_code=500, 
            detail=f"An internal error occurred in the prediction logic: {str(e)}"
        )