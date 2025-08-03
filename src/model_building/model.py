# model.py

import numpy as np
from typing import Optional, Literal


class LinearRegressionScratch:
    """
    Implements Linear Regression using:
    1. Normal Equation (closed-form)
    2. Gradient Descent (iterative)
    """

    def __init__(self) -> None:
        self.weights: Optional[np.ndarray] = None

    def add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add bias term (column of ones) to the input features."""
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            method: Literal["normal", "gradientDescent"],
            learning_rate: float = 0.01,
            n_iterations: int = 1000) -> None:
        """
        Train the linear regression model.
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            method: "normal" or "gradientDescent"
            learning_rate: Step size for gradient descent
            n_iterations: Number of iterations for gradient descent
        """
        X_b = self.add_bias(X)
        n_samples, n_features = X_b.shape

        if method == "normal":
            # Closed-form: (XᵀX)⁻¹Xᵀy
            XT_X = X_b.T.dot(X_b)
            XT_y = X_b.T.dot(y)
            self.weights = np.linalg.inv(XT_X).dot(XT_y)

        elif method == "gradientDescent":
            # Iterative solution
            self.weights = np.zeros(n_features)
            for _ in range(n_iterations):
                y_pred = X_b.dot(self.weights)
                error = y_pred - y
                gradient = (2 / n_samples) * X_b.T.dot(error)
                self.weights -= learning_rate * gradient

        else:
            raise ValueError("Method must be either 'normal' or 'gradientDescent'")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if self.weights is None:
            raise ValueError("Model is not trained yet.")
        X_b = self.add_bias(X)
        return X_b.dot(self.weights)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score."""
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
