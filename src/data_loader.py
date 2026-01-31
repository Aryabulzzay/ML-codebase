import pandas as pd
import numpy as np
from typing import Tuple, Optional

class DataLoader:
    """Load and validate dataset for ML tasks."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None
    
    def load_csv(self) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            self.data = pd.read_csv(self.filepath)
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.filepath}")
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    def get_feature_target(self, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Split data into features and target."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_csv() first.")
        
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not in data.")
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return X, y
    
    def get_data_info(self) -> dict:
        """Get basic information about the dataset."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_csv() first.")
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': self.data.dtypes.to_dict()
        }
