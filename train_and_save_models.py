import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

DATAPATH = 'app/dataset/electricity_cost_dataset.csv'
TARGET_COL = 'electricity cost'
filename = 'res'

def prepare_type_data(data_path=DATAPATH):
    df = pd.read_csv(data_path)

    # Global Cleaning
    df = df.drop(['recycling rate', 'air qality index', 'issue reolution time'], axis=1, errors='ignore')

    # Filer for residential and clean up
    res_df = df[df['structure type'] == 'Residential'].copy()
    res_df = res_df.drop('structure type', axis=1)

    print("Type Data Prepared. Features: ", res_df.columns.drop('electricity cost').tolist())
    return res_df

def scale_features(data: np.ndarray):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoiding division by 0
    range_vals = max_vals - min_vals
    
    if np.isscalar(range_vals):
        if range_vals == 0:
            range_vals = 1
    else:
        range_vals[range_vals == 0] = 1

    scaled_data = (data - min_vals) / range_vals

    return scaled_data, {'min': min_vals, 'max': max_vals}

def inverse_scale_target(scaled_target: np.ndarray, y_scaler_params: dict):
    return scaled_target * (y_scaler_params['max'] - y_scaler_params['min']) + y_scaler_params['min']

def run_trainin_pipeline(df: pd.DataFrame, target_col: str, model_params: dict):
    print(f'\n{'='*10} RUNNING TRAINING {'='*10}')

    # 1. Separate features and target, then split the data
    X_raw = df.drop(target_col, axis=1).values
    y_raw = df[target_col].values

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)


    # 2. Data Scaling
    X_train_scaled, x_scaler_params = scale_features(X_train_raw)
    y_train_scaled, y_scaler_params = scale_features(y_train_raw)

    # Apply same scaling to test data
    X_test_scaled = (X_test_raw - x_scaler_params['min']) / (x_scaler_params['max'] - x_scaler_params['min'])

    # 3. Train the model
    print(f'Training model with params: {model_params}')
    model = LinearRegressionGD(**model_params)
    model.fit(X_train_scaled, y_train_scaled)
    print(f'Training Complete.')


    # 4. Evaluate the model on the TEST Set
    print("\n--- Model Evaluation ---")
    y_pred_scaled = model.predict(X_test_scaled)

    # Inverse Transform the predictions to get them in the original scale
    y_pred_raw = inverse_scale_target(y_pred_scaled, y_scaler_params)

    # Calculating MSE on the original scale
    mse = np.mean((y_test_raw - y_pred_raw.flatten()) ** 2)
    print(f'Mean Squared Error (MSE) on Test Set: {mse:,.2f}')

    # Plot cost history
    # plt.figure(figsize=(10,5))
    # plt.plot(range(len(model.cost_history)), model.cost_history)
    # plt.title(f"Cost History (lr={model.learning_rate})")
    # plt.xlabel("Iteration")
    # plt.ylabel("Cost (MSE on scaled data)")
    # plt.grid(True)
    # plt.show()

    return model, x_scaler_params, y_scaler_params, mse

# --- Execute the pipeline for the Residential Data ---
if __name__ == '__main__':
    residential_def = prepare_type_data()

    print("\n -- Searching for the best hyperparameters --")
    learning_rates = [0.3, 0.1, 0.03, 0.01]
    iterations_list =[5000, 10000, 20000]

    best_mse = float('inf')
    best_params = {}
    best_artifacts = {}

    for lr in learning_rates:
        for n_iter in iterations_list:
            params = {'learning_rate': lr, 'n_iterations': n_iter}

            model, x_scaler, y_scaler, mse = run_trainin_pipeline(
                df=residential_def,
                target_col=TARGET_COL,
                model_params=params
            )

            if mse < best_mse:
                print(f"New Best MSE found: {mse:,.2f} with params {params}")
                best_mse = mse
                best_params = params
                best_artifacts = {
                    'model': model,
                    'x_scaler': x_scaler,
                    'y_scaler': y_scaler
                }

    print("\n" + "="*50)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"    Best MSE Found: {best_mse:,.2f}")
    print(f"    Best Parameters: {best_params}")
    print("="*50)

    scaler_to_save = {
        'x_scaler_params': best_artifacts['x_scaler'],
        'y_scaler_params': best_artifacts['y_scaler']
    }
    joblib.dump(best_artifacts['model'], f'app/models/{filename}_model.joblib')
    joblib.dump(scaler_to_save, f'app/models/{filename}_scalers.joblib')
    print(f"Models are saved in the models directory")

