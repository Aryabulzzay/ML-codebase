import pytest
import pandas as pd
import numpy as np
from unittest.mock import mock_open, patch
import tempfile
import os

from src.data_loader import DataLoader

class TestDataLoader:
    """Test suite for DataLoader class."""
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader("dummy.csv")
        assert loader.filepath == "dummy.csv"
        assert loader.data is None
    
    def test_load_csv_success(self):
        """Test successful CSV loading."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,col3\n1,2,3\n4,5,6\n7,8,9")
            temp_file = f.name
        
        try:
            loader = DataLoader(temp_file)
            data = loader.load_csv()
            
            assert isinstance(data, pd.DataFrame)
            assert data.shape == (3, 3)
            assert list(data.columns) == ['col1', 'col2', 'col3']
            assert loader.data is not None
        finally:
            os.unlink(temp_file)
    
    def test_load_csv_file_not_found(self):
        """Test CSV loading with non-existent file."""
        loader = DataLoader("non_existent.csv")
        
        with pytest.raises(FileNotFoundError):
            loader.load_csv()
    
    def test_get_feature_target_success(self):
        """Test successful feature-target split."""
        # Create sample data
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        
        loader = DataLoader("dummy.csv")
        loader.data = data
        
        X, y = loader.get_feature_target('target')
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape == (3, 2)
        assert len(y) == 3
        assert 'target' not in X.columns
        assert list(X.columns) == ['feature1', 'feature2']
    
    def test_get_feature_target_no_data(self):
        """Test feature-target split without loaded data."""
        loader = DataLoader("dummy.csv")
        
        with pytest.raises(ValueError, match="Data not loaded"):
            loader.get_feature_target('target')
    
    def test_get_feature_target_invalid_column(self):
        """Test feature-target split with invalid target column."""
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        
        loader = DataLoader("dummy.csv")
        loader.data = data
        
        with pytest.raises(ValueError, match="Target column 'invalid' not in data"):
            loader.get_feature_target('invalid')
    
    def test_get_data_info_success(self):
        """Test successful data info retrieval."""
        data = pd.DataFrame({
            'A': [1, 2, None],
            'B': ['x', 'y', 'z']
        })
        
        loader = DataLoader("dummy.csv")
        loader.data = data
        
        info = loader.get_data_info()
        
        assert isinstance(info, dict)
        assert info['shape'] == (3, 2)
        assert info['columns'] == ['A', 'B']
        assert info['missing_values']['A'] == 1
        assert info['missing_values']['B'] == 0
        assert 'float64' in str(info['dtypes']['A'])
        assert str(info['dtypes']['B']) == 'str'
    
    def test_get_data_info_no_data(self):
        """Test data info retrieval without loaded data."""
        loader = DataLoader("dummy.csv")
        
        with pytest.raises(ValueError, match="Data not loaded"):
            loader.get_data_info()