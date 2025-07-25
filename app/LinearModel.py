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

    def mse_score(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - (ss_res / ss_tot)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on test data and returns the MSE on the original scale.
        """
        predictions = self.predict(X_test) 
        y_test_flat = y_test.flatten()
        predictions_flat = predictions.flatten()
        
        mse = np.mean((predictions_flat - y_test_flat)**2)
        return mse
