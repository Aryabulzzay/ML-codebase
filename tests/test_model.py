import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from src.model import SimpleLinearRegression

class TestSimpleLinearRegression:
    """Test suite for SimpleLinearRegression class."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
        self.X_train = X[:80]
        self.y_train = y[:80]
        self.X_test = X[80:]
        self.y_test = y[80:]
        self.model = SimpleLinearRegression(learning_rate=0.01, n_iterations=1000)
    
    def test_init(self):
        """Test SimpleLinearRegression initialization."""
        assert self.model.learning_rate == 0.01
        assert self.model.n_iterations == 1000
        assert self.model.weights is None
        assert self.model.bias is None
        assert len(self.model.loss_history) == 0
    
    def test_init_invalid_learning_rate(self):
        """Test initialization with invalid learning rate."""
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            SimpleLinearRegression(learning_rate=0)
        
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            SimpleLinearRegression(learning_rate=-0.01)
    
    def test_init_invalid_iterations(self):
        """Test initialization with invalid iterations."""
        with pytest.raises(ValueError, match="Number of iterations must be positive"):
            SimpleLinearRegression(n_iterations=0)
    
    def test_fit(self):
        """Test model fitting."""
        model = self.model.fit(self.X_train, self.y_train)
        
        assert model.weights is not None
        assert model.bias is not None
        assert len(model.weights) == self.X_train.shape[1]
        assert len(model.loss_history) == self.model.n_iterations
        
        # Loss should decrease over time
        assert model.loss_history[0] > model.loss_history[-1]
    
    def test_fit_mismatched_shapes(self):
        """Test fitting with mismatched X and y shapes."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="X and y must have same length"):
            self.model.fit(X, y)
    
    def test_predict(self):
        """Test model predictions."""
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (self.X_test.shape[0],)
        
        # Predictions should be close to actual values (within reasonable range)
        mse = np.mean((predictions - self.y_test) ** 2)
        assert mse < 100  # Reasonable threshold for synthetic data
    
    def test_predict_not_trained(self):
        """Test predictions without training."""
        with pytest.raises(ValueError, match="Model not trained"):
            self.model.predict(self.X_test)
    
    def test_evaluate(self):
        """Test model evaluation."""
        self.model.fit(self.X_train, self.y_train)
        metrics = self.model.evaluate(self.X_test, self.y_test)
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['r2'] <= 1  # R² can be negative for bad models
        
        # For a reasonably good model on synthetic data
        assert metrics['mse'] < 100
        assert metrics['r2'] > 0.5
    
    def test_get_params(self):
        """Test getting model parameters."""
        params = self.model.get_params()
        
        assert isinstance(params, dict)
        assert params['learning_rate'] == 0.01
        assert params['n_iterations'] == 1000
        assert params['weights'] is None
        assert params['bias'] is None
        
        # Test after fitting
        self.model.fit(self.X_train, self.y_train)
        params = self.model.get_params()
        
        assert params['weights'] is not None
        assert params['bias'] is not None
        assert len(params['weights']) == self.X_train.shape[1]
    
    def test_compute_loss(self):
        """Test loss computation."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 2.9])
        
        loss = self.model._compute_loss(y_true, y_pred)
        
        assert isinstance(loss, float)
        assert loss >= 0
        
        # Perfect prediction should have zero loss
        perfect_loss = self.model._compute_loss(y_true, y_true)
        assert perfect_loss == 0.0
    
    def test_initialize_parameters(self):
        """Test parameter initialization."""
        self.model._initialize_parameters(5)
        
        assert self.model.weights is not None
        assert len(self.model.weights) == 5
        assert isinstance(self.model.bias, (int, float))