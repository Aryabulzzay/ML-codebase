import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.preprocessing import DataPreprocessor

class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    def setup_method(self):
        """Setup test data."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data with missing values
        self.df = pd.DataFrame({
            'numeric1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'numeric2': [10.0, 20.0, 30.0, np.nan, 50.0],
            'categorical': ['A', 'B', 'A', 'B', 'A'],
            'text': ['cat', 'dog', 'cat', 'dog', 'bird']
        })
    
    def test_init(self):
        """Test DataPreprocessor initialization."""
        assert isinstance(self.preprocessor.scaler, StandardScaler)
        assert not self.preprocessor.is_fitted
    
    def test_handle_missing_values_mean(self):
        """Test missing value handling with mean strategy."""
        df_clean = self.preprocessor.handle_missing_values(self.df, 'mean')
        
        assert not df_clean.isnull().any().any()
        assert df_clean['numeric1'].mean() == pytest.approx(3.0)
        assert df_clean['numeric2'].mean() == pytest.approx(27.5)
    
    def test_handle_missing_values_median(self):
        """Test missing value handling with median strategy."""
        df_clean = self.preprocessor.handle_missing_values(self.df, 'median')
        
        assert not df_clean.isnull().any().any()
        assert df_clean['numeric1'].median() == pytest.approx(3.0)
        assert df_clean['numeric2'].median() == pytest.approx(25.0)
    
    def test_handle_missing_values_mode(self):
        """Test missing value handling with mode strategy."""
        df_clean = self.preprocessor.handle_missing_values(self.df, 'mode')
        
        assert not df_clean.isnull().any().any()
        assert df_clean['numeric1'].mode()[0] == pytest.approx(1.0)
        assert df_clean['numeric2'].mode()[0] == pytest.approx(10.0)
    
    def test_handle_missing_values_drop(self):
        """Test missing value handling with drop strategy."""
        df_clean = self.preprocessor.handle_missing_values(self.df, 'drop')
        
        assert not df_clean.isnull().any().any()
        assert df_clean.shape[0] == 3  # Rows with missing values dropped
    
    def test_handle_missing_values_invalid_strategy(self):
        """Test missing value handling with invalid strategy."""
        with pytest.raises(ValueError, match="Strategy must be"):
            self.preprocessor.handle_missing_values(self.df, 'invalid')
    
    def test_encode_categorical_success(self):
        """Test categorical encoding."""
        df_encoded = self.preprocessor.encode_categorical(
            self.df[['categorical', 'text']], 
            ['categorical', 'text']
        )
        
        assert df_encoded['categorical'].dtype in [np.int64, np.int32]
        assert df_encoded['text'].dtype in [np.int64, np.int32]
        assert len(set(df_encoded['categorical'])) == 2
        assert len(set(df_encoded['text'])) == 3
    
    def test_encode_categorical_column_not_found(self):
        """Test categorical encoding with non-existent column."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            self.preprocessor.encode_categorical(self.df, ['nonexistent'])
    
    def test_encode_categorical_non_categorical(self):
        """Test categorical encoding with non-categorical column."""
        with pytest.raises(ValueError, match="Column 'numeric1' is not categorical"):
            self.preprocessor.encode_categorical(self.df, ['numeric1'])
    
    def test_scale_features_fit(self):
        """Test feature scaling with fitting."""
        numeric_df = self.df[['numeric1', 'numeric2']].fillna(0)
        df_scaled = self.preprocessor.scale_features(numeric_df, fit=True)
        
        assert self.preprocessor.is_fitted
        assert df_scaled['numeric1'].mean() == pytest.approx(0.0, abs=1e-10)
        assert df_scaled['numeric1'].std(ddof=0) == pytest.approx(1.0, abs=1e-10)
    
    def test_scale_features_no_fit(self):
        """Test feature scaling without prior fitting."""
        numeric_df = self.df[['numeric1', 'numeric2']].fillna(0)
        
        with pytest.raises(ValueError, match="Scaler not fitted"):
            self.preprocessor.scale_features(numeric_df, fit=False)
    
    def test_scale_features_no_numeric(self):
        """Test feature scaling with no numeric columns."""
        categorical_df = self.df[['categorical', 'text']]
        df_scaled = self.preprocessor.scale_features(categorical_df)
        
        # Should return unchanged DataFrame
        pd.testing.assert_frame_equal(df_scaled, categorical_df)
    
    def test_train_test_split_custom(self):
        """Test custom train-test split."""
        X = pd.DataFrame({'feature': range(100)})
        y = pd.Series(range(100))
        
        X_train, X_test, y_train, y_test = self.preprocessor.train_test_split_custom(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert len(pd.concat([X_train, X_test])) == 100
    
    def test_train_test_split_invalid_test_size(self):
        """Test train-test split with invalid test size."""
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([1, 2, 3])
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            self.preprocessor.train_test_split_custom(X, y, test_size=1.5)
    
    def test_train_test_split_mismatched_lengths(self):
        """Test train-test split with mismatched X and y lengths."""
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([1, 2])
        
        with pytest.raises(ValueError, match="X and y must have same length"):
            self.preprocessor.train_test_split_custom(X, y)