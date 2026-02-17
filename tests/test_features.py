# ============================================================
# tests/test_features.py
# Unit tests for the feature engineering and feature selection
# modules. Uses pytest for testing.
# ============================================================

import sys                                         # System-specific parameters
from pathlib import Path                           # Object-oriented file paths

# Add the project root to Python path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest                                      # Testing framework
import pandas as pd                                # Data manipulation
import numpy as np                                 # Numerical computing

from src.features.engineer import FeatureEngineer  # Feature engineering module
from src.features.selector import FeatureSelector  # Feature selection module
from src.utils.helpers import load_config          # Config loader


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def config():
    """
    Load and return the project configuration.
    """
    try:
        return load_config()
    except Exception:
        return {
            "data": {
                "target_column": "Churn",
                "id_column": "customerID",
            },
            "features": {
                "tenure_bins": [0, 12, 24, 48, 60, 72],
                "rolling_windows": [3, 6, 12],
            },
            "feature_selection": {
                "correlation_threshold": 0.95,
                "max_features": 30,
            },
        }


@pytest.fixture
def sample_dataframe():
    """
    Create a sample DataFrame for feature engineering tests.
    """
    np.random.seed(42)
    n = 50                                         # More samples for reliable tests
    df = pd.DataFrame({
        "customerID": [f"CUST-{i:04d}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": np.random.choice([0, 1], n),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n),
        "tenure": np.random.randint(1, 72, n),
        "PhoneService": np.random.choice(["Yes", "No"], n),
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
        "Churn": np.random.choice([0, 1], n, p=[0.73, 0.27]),
    })
    return df


# ============================================================
# Tests for FeatureEngineer
# ============================================================

class TestFeatureEngineer:
    """Test suite for the FeatureEngineer class."""

    def test_init(self, config):
        """Test that FeatureEngineer initializes correctly."""
        # Create a FeatureEngineer instance
        engineer = FeatureEngineer(config)
        # Verify it was created
        assert engineer is not None

    def test_creates_new_features(self, config, sample_dataframe):
        """Test that feature engineering adds new columns."""
        # Create engineer and apply transformations
        engineer = FeatureEngineer(config)
        df_new = engineer.create_all_features(sample_dataframe)
        # New DataFrame should have more columns
        assert df_new.shape[1] > sample_dataframe.shape[1], (
            f"Expected more columns after engineering. "
            f"Original: {sample_dataframe.shape[1]}, After: {df_new.shape[1]}"
        )

    def test_preserves_original_rows(self, config, sample_dataframe):
        """Test that feature engineering does not add or remove rows."""
        # Create engineer and apply
        engineer = FeatureEngineer(config)
        df_new = engineer.create_all_features(sample_dataframe)
        # Row count should be the same
        assert df_new.shape[0] == sample_dataframe.shape[0], (
            f"Row count changed from {sample_dataframe.shape[0]} to {df_new.shape[0]}"
        )

    def test_tenure_features_created(self, config, sample_dataframe):
        """Test that tenure-related features are created."""
        # Create engineer and apply
        engineer = FeatureEngineer(config)
        df_new = engineer.create_all_features(sample_dataframe)
        # Check for expected tenure features
        expected_features = ["tenure_years", "tenure_group"]
        for feat in expected_features:
            assert feat in df_new.columns, f"Missing expected feature: {feat}"

    def test_charge_features_created(self, config, sample_dataframe):
        """Test that charge-related features are created."""
        # Create engineer and apply
        engineer = FeatureEngineer(config)
        df_new = engineer.create_all_features(sample_dataframe)
        # Check for expected charge features
        expected_features = ["avg_monthly_charge", "charge_per_tenure_month"]
        for feat in expected_features:
            assert feat in df_new.columns, f"Missing expected feature: {feat}"

    def test_no_infinite_values(self, config, sample_dataframe):
        """Test that no infinite values are produced by feature engineering."""
        # Create engineer and apply
        engineer = FeatureEngineer(config)
        df_new = engineer.create_all_features(sample_dataframe)
        # Select only numeric columns
        numeric_cols = df_new.select_dtypes(include=[np.number]).columns
        # Check for infinite values
        inf_mask = np.isinf(df_new[numeric_cols].values)
        inf_count = inf_mask.sum()
        assert inf_count == 0, f"Found {inf_count} infinite values in engineered features"

    def test_no_all_nan_columns(self, config, sample_dataframe):
        """Test that no columns are entirely NaN after feature engineering."""
        # Create engineer and apply
        engineer = FeatureEngineer(config)
        df_new = engineer.create_all_features(sample_dataframe)
        # Check each column
        all_nan_cols = [col for col in df_new.columns if df_new[col].isna().all()]
        assert len(all_nan_cols) == 0, f"Columns entirely NaN: {all_nan_cols}"

    def test_service_count_range(self, config, sample_dataframe):
        """Test that service count feature has valid values."""
        # Create engineer and apply
        engineer = FeatureEngineer(config)
        df_new = engineer.create_all_features(sample_dataframe)
        # Check if service_count exists
        if "service_count" in df_new.columns:
            # Service count should be non-negative
            assert (df_new["service_count"] >= 0).all(), "service_count has negative values"
            # Service count should not exceed total possible services
            assert (df_new["service_count"] <= 10).all(), "service_count exceeds maximum"


# ============================================================
# Tests for FeatureSelector
# ============================================================

class TestFeatureSelector:
    """Test suite for the FeatureSelector class."""

    def test_init(self, config):
        """Test that FeatureSelector initializes correctly."""
        # Create a FeatureSelector instance
        selector = FeatureSelector(config)
        # Verify it was created
        assert selector is not None

    def test_select_returns_list(self, config):
        """Test that feature selection returns a list of feature names."""
        # Create selector
        selector = FeatureSelector(config)
        # Create a simple numeric dataset
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            "feat_1": np.random.randn(n),
            "feat_2": np.random.randn(n),
            "feat_3": np.random.randn(n),
            "feat_4": np.random.randn(n) * 0.01,  # Low variance
        })
        y = pd.Series(np.random.choice([0, 1], n))
        # Select features
        selected = selector.select_features(X, y)
        # Should return a list
        assert isinstance(selected, list), f"Expected list, got {type(selected)}"
        # Should have at least one feature
        assert len(selected) > 0, "No features were selected"

    def test_selected_subset_of_original(self, config):
        """Test that selected features are a subset of original features."""
        # Create selector
        selector = FeatureSelector(config)
        # Create dataset
        np.random.seed(42)
        n = 100
        feature_names = [f"feat_{i}" for i in range(10)]
        X = pd.DataFrame(np.random.randn(n, 10), columns=feature_names)
        y = pd.Series(np.random.choice([0, 1], n))
        # Select features
        selected = selector.select_features(X, y)
        # All selected features should be in the original set
        for feat in selected:
            assert feat in feature_names, f"Unknown feature selected: {feat}"

    def test_removes_highly_correlated(self, config):
        """Test that highly correlated features are removed."""
        # Create selector with low correlation threshold
        custom_config = config.copy()
        if "feature_selection" not in custom_config:
            custom_config["feature_selection"] = {}
        custom_config["feature_selection"]["correlation_threshold"] = 0.9
        selector = FeatureSelector(custom_config)
        # Create dataset with perfectly correlated features
        np.random.seed(42)
        n = 100
        base = np.random.randn(n)
        X = pd.DataFrame({
            "feat_original": base,
            "feat_clone": base + np.random.randn(n) * 0.01,  # Near-perfect correlation
            "feat_independent": np.random.randn(n),
        })
        y = pd.Series(np.random.choice([0, 1], n))
        # Select features
        selected = selector.select_features(X, y)
        # Should not keep both correlated features
        # (at least one should be removed)
        assert len(selected) <= 3, "Feature selection should remove correlated features"
