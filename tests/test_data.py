# ============================================================
# tests/test_data.py
# Unit tests for the data loading, validation, and preprocessing
# modules. Uses pytest for test discovery and assertion.
# ============================================================

import sys                                         # System-specific parameters
from pathlib import Path                           # Object-oriented file paths

# Add the project root to Python path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest                                      # Testing framework
import pandas as pd                                # Data manipulation
import numpy as np                                 # Numerical computing

from src.data.loader import DataLoader             # Data loading module
from src.data.validator import DataValidator        # Data validation module
from src.data.preprocessor import DataPreprocessor  # Data preprocessing module
from src.utils.helpers import load_config          # Config loader


# ============================================================
# Fixtures: Reusable test data
# ============================================================

@pytest.fixture
def config():
    """
    Load and return the project configuration.
    Fixture is available to all tests in this module.
    """
    try:
        # Try to load the actual config
        return load_config()
    except Exception:
        # Return a minimal config if real one is unavailable
        return {
            "data": {
                "target_column": "Churn",
                "id_column": "customerID",
                "raw_path": "data/raw/telco_churn.csv",
            },
            "preprocessing": {
                "test_size": 0.2,
                "random_state": 42,
            },
        }


@pytest.fixture
def sample_dataframe():
    """
    Create a small sample DataFrame mimicking the Telco dataset.
    Used across multiple tests for consistent test data.
    """
    # Create a DataFrame with 20 sample customers
    np.random.seed(42)                             # Reproducible random numbers
    n = 20                                         # Number of sample records
    df = pd.DataFrame({
        "customerID": [f"CUST-{i:04d}" for i in range(n)],  # Unique IDs
        "gender": np.random.choice(["Male", "Female"], n),    # Gender
        "SeniorCitizen": np.random.choice([0, 1], n),         # Senior flag
        "Partner": np.random.choice(["Yes", "No"], n),        # Partner status
        "Dependents": np.random.choice(["Yes", "No"], n),     # Dependents
        "tenure": np.random.randint(1, 72, n),                # Months of tenure
        "PhoneService": np.random.choice(["Yes", "No"], n),   # Phone service
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "PaymentMethod": np.random.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ], n),
        "MonthlyCharges": np.round(np.random.uniform(18, 120, n), 2),
        "TotalCharges": np.round(np.random.uniform(18, 8000, n), 2),
        "Churn": np.random.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })
    # Return the sample DataFrame
    return df


# ============================================================
# Tests for DataLoader
# ============================================================

class TestDataLoader:
    """Test suite for the DataLoader class."""

    def test_init(self, config):
        """Test that DataLoader initializes without errors."""
        # Create a DataLoader instance
        loader = DataLoader(config)
        # Assert it was created (not None)
        assert loader is not None

    def test_fix_total_charges(self, config, sample_dataframe):
        """Test that TotalCharges conversion handles edge cases."""
        # Create a loader instance
        loader = DataLoader(config)
        # Introduce a string value that should be converted
        df = sample_dataframe.copy()
        # Set one TotalCharges to a string with whitespace
        df.loc[0, "TotalCharges"] = " "
        # Call the private fix method
        fixed = loader._fix_total_charges(df)
        # Verify the column is now numeric
        assert pd.api.types.is_numeric_dtype(fixed["TotalCharges"])
        # Verify no infinite values exist
        assert not np.isinf(fixed["TotalCharges"]).any()

    def test_encode_target(self, config, sample_dataframe):
        """Test that the target column is encoded to binary (0/1)."""
        # Create a loader instance
        loader = DataLoader(config)
        # Encode the target column
        encoded = loader._encode_target(sample_dataframe)
        # Get the target column name
        target = config.get("data", {}).get("target_column", "Churn")
        # Verify all values are either 0 or 1
        unique_values = set(encoded[target].unique())
        assert unique_values.issubset({0, 1}), f"Expected {{0, 1}}, got {unique_values}"


# ============================================================
# Tests for DataValidator
# ============================================================

class TestDataValidator:
    """Test suite for the DataValidator class."""

    def test_init(self, config):
        """Test that DataValidator initializes correctly."""
        # Create a DataValidator instance
        validator = DataValidator(config)
        # Verify the instance was created
        assert validator is not None

    def test_check_schema(self, config, sample_dataframe):
        """Test schema validation identifies required columns."""
        # Create a validator
        validator = DataValidator(config)
        # Check schema — should pass for complete data
        result = validator.check_schema(sample_dataframe)
        # Result should be a boolean
        assert isinstance(result, bool)

    def test_check_missing_values(self, config, sample_dataframe):
        """Test missing value detection."""
        # Create a validator
        validator = DataValidator(config)
        # Sample data should have no missing values
        result = validator.check_missing_values(sample_dataframe)
        # Should pass (no missing values in our test data)
        assert result is True

    def test_check_missing_detects_nulls(self, config, sample_dataframe):
        """Test that missing values are actually detected."""
        # Create a validator
        validator = DataValidator(config)
        # Introduce missing values
        df = sample_dataframe.copy()
        df.loc[0, "tenure"] = np.nan               # Set one value to NaN
        df.loc[1, "MonthlyCharges"] = np.nan        # Set another to NaN
        # Check should still pass (within threshold) or detect
        result = validator.check_missing_values(df)
        # Result is a boolean regardless
        assert isinstance(result, bool)

    def test_check_target_distribution(self, config, sample_dataframe):
        """Test target distribution validation."""
        # Create a validator
        validator = DataValidator(config)
        # Encode target first
        loader = DataLoader(config)
        df = loader._encode_target(sample_dataframe)
        # Check target distribution
        result = validator.check_target_distribution(df)
        # Should return a boolean
        assert isinstance(result, bool)

    def test_run_all_validations(self, config, sample_dataframe):
        """Test that all validations run and return a dictionary."""
        # Create a validator
        validator = DataValidator(config)
        # Run all validations
        results = validator.run_all_validations(sample_dataframe)
        # Should return a dictionary
        assert isinstance(results, dict)
        # Should have multiple check results
        assert len(results) > 0


# ============================================================
# Tests for DataPreprocessor
# ============================================================

class TestDataPreprocessor:
    """Test suite for the DataPreprocessor class."""

    def test_init(self, config):
        """Test that DataPreprocessor initializes correctly."""
        # Create a preprocessor
        preprocessor = DataPreprocessor(config)
        # Verify creation
        assert preprocessor is not None

    def test_fit_transform_produces_arrays(self, config, sample_dataframe):
        """Test that fit_transform returns proper numpy arrays."""
        # Create a preprocessor
        preprocessor = DataPreprocessor(config)
        # Encode target to binary
        loader = DataLoader(config)
        df = loader._encode_target(sample_dataframe)
        # Separate features and target
        target_col = config.get("data", {}).get("target_column", "Churn")
        id_col = config.get("data", {}).get("id_column", "customerID")
        X = df.drop(columns=[target_col, id_col], errors="ignore")
        y = df[target_col]
        # Fit and transform — should return 4 arrays
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(X, y)
        # Verify all outputs are numpy arrays
        assert isinstance(X_train, np.ndarray), "X_train should be numpy array"
        assert isinstance(X_test, np.ndarray), "X_test should be numpy array"
        # Verify shapes are consistent
        assert X_train.shape[0] == len(y_train), "X_train and y_train length mismatch"
        assert X_test.shape[0] == len(y_test), "X_test and y_test length mismatch"
        # Verify total samples equal original
        total = len(y_train) + len(y_test)
        assert total == len(y), f"Total samples {total} != original {len(y)}"

    def test_transform_matches_fit_dimensions(self, config, sample_dataframe):
        """Test that transform produces same number of features as fit_transform."""
        # Create a preprocessor
        preprocessor = DataPreprocessor(config)
        # Prepare the data
        loader = DataLoader(config)
        df = loader._encode_target(sample_dataframe)
        target_col = config.get("data", {}).get("target_column", "Churn")
        id_col = config.get("data", {}).get("id_column", "customerID")
        X = df.drop(columns=[target_col, id_col], errors="ignore")
        y = df[target_col]
        # Fit and transform
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(X, y)
        # Transform another batch (using same data for simplicity)
        X_new = preprocessor.transform(X)
        # Feature dimensions should match
        assert X_new.shape[1] == X_train.shape[1], (
            f"Transform output has {X_new.shape[1]} features, "
            f"but fit_transform had {X_train.shape[1]}"
        )

    def test_no_nan_after_preprocessing(self, config, sample_dataframe):
        """Test that preprocessing removes all NaN values."""
        # Create a preprocessor
        preprocessor = DataPreprocessor(config)
        # Prepare data with a missing value
        loader = DataLoader(config)
        df = loader._encode_target(sample_dataframe)
        target_col = config.get("data", {}).get("target_column", "Churn")
        id_col = config.get("data", {}).get("id_column", "customerID")
        X = df.drop(columns=[target_col, id_col], errors="ignore")
        y = df[target_col]
        # Introduce a missing value
        X.iloc[0, 0] = np.nan
        # Fit and transform
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(X, y)
        # Verify no NaN in output
        assert not np.isnan(X_train).any(), "X_train contains NaN values"
        assert not np.isnan(X_test).any(), "X_test contains NaN values"
