import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator

class SimpleLinearRegression:
    """Simple linear regression model for demonstration."""
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if n_iterations <= 0:
            raise ValueError("Number of iterations must be positive")
        
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def _initialize_parameters(self, n_features: int):
        """Initialize model parameters."""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error loss."""
        return np.mean((y_true - y_pred) ** 2)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleLinearRegression':
        """Train the linear regression model."""
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        
        for _ in range(self.n_iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Backward pass (gradients)
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.weights is None or self.bias is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return np.dot(X, self.weights) + self.bias
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        mse = self._compute_loss(y, y_pred)
        mae = np.mean(np.abs(y - y_pred))
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2)
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias
        }