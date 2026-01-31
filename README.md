# ML Codebase

A machine learning project demonstrating data loading, preprocessing, and a simple linear regression model implementation.

## Features

- **Data Loading**: CSV data loading with validation
- **Preprocessing**: Missing value handling, categorical encoding, feature scaling, and train/test splitting
- **Model**: Custom implementation of simple linear regression with gradient descent

## Project Structure

```
.
├── src/
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessing.py    # Data preprocessing functions
│   └── model.py           # Linear regression model
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_model.py
├── requirements.txt       # Python dependencies
├── pytest.ini            # Test configuration
└── README.md             # This file
```

## Installation

1. Clone or download the project
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

This project uses pytest for testing. To run the test suite:

### Basic test run:
```bash
pytest
```

### Run with verbose output:
```bash
pytest -v
```

### Run specific test file:
```bash
pytest tests/test_model.py
```

### Run specific test:
```bash
pytest tests/test_model.py::TestSimpleLinearRegression::test_fit -v
```

### Generate coverage report:
```bash
pytest --cov=src --cov-report=html
```

The test configuration in `pytest.ini` includes:
- Coverage reporting (HTML and XML)
- Verbose output by default
- Test discovery from `tests/` directory

## Test Coverage

The project maintains high test coverage (>95%) across all modules:
- `data_loader.py`: Data loading and validation
- `model.py`: Linear regression implementation
- `preprocessing.py`: Data preprocessing utilities

## Usage

After running tests successfully, you can use the modules in your own projects:

```python
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.model import SimpleLinearRegression

# Load data
loader = DataLoader('path/to/data.csv')
data = loader.load_csv()

# Preprocess
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.train_test_split(data, 'target_column')

# Train model
model = SimpleLinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## Dependencies

- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- coverage >= 6.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- black >= 22.0.0 (optional, for code formatting)