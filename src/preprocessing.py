import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional

class DataPreprocessor:
    """Preprocess data for ML models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Handle missing values in DataFrame."""
        if strategy not in ['mean', 'median', 'mode', 'drop']:
            raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
        
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy == 'mean':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif strategy == 'mode':
            for col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0)
        
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Encode categorical columns using LabelEncoder."""
        df_encoded = df.copy()
        
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            if not isinstance(df[col].dtype, pd.CategoricalDtype) and not isinstance(df[col].dtype, pd.StringDtype) and df[col].dtype != 'object':
                raise ValueError(f"Column '{col}' is not categorical")
            
            df_encoded[col] = self.label_encoder.fit_transform(df[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df
        
        df_scaled = df.copy()
        
        if fit:
            df_scaled[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df_scaled[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df_scaled
    
    def train_test_split_custom(self, X: pd.DataFrame, y: pd.Series, 
                               test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Custom train-test split for demonstration."""
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        test_size_int = int(len(X) * test_size)
        
        test_idx = indices[:test_size_int]
        train_idx = indices[test_size_int:]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        return X_train, X_test, y_train, y_test