# ============================================================
# tests/test_models.py
# Unit tests for the model training, evaluation, explainability,
# and registry modules. Uses pytest for testing.
# ============================================================

import sys                                         # System-specific parameters
from pathlib import Path                           # Object-oriented file paths

# Add the project root to Python path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest                                      # Testing framework
import pandas as pd                                # Data manipulation
import numpy as np                                 # Numerical computing
import tempfile                                    # Temporary file handling
import shutil                                      # File/directory operations

from src.models.trainer import ModelTrainer        # Model training module
from src.models.evaluator import ModelEvaluator    # Model evaluation module
from src.models.registry import ModelRegistry      # Model registry module
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
            "data": {"target_column": "Churn"},
            "models": {
                "algorithms": ["logistic_regression", "random_forest"],
                "logistic_regression": {"C": 1.0, "max_iter": 100},
                "random_forest": {"n_estimators": 50, "max_depth": 5},
                "cross_validation": {"n_folds": 3},
                "calibration": {"method": "isotonic", "cv": 3},
                "threshold_optimization": {
                    "retention_cost": 50,
                    "customer_value": 500,
                },
            },
            "model_registry": {"dir": "test_registry"},
        }


@pytest.fixture
def binary_dataset():
    """
    Create a synthetic binary classification dataset.
    Returns X_train, X_test, y_train, y_test as numpy arrays.
    """
    np.random.seed(42)                             # Reproducible random numbers
    n_train = 200                                  # Training samples
    n_test = 50                                    # Test samples
    n_features = 10                                # Number of features

    # Generate training data with some signal
    X_train = np.random.randn(n_train, n_features)
    # Create target correlated with first feature
    y_train = (X_train[:, 0] + X_train[:, 1] + np.random.randn(n_train) * 0.5 > 0).astype(int)

    # Generate test data the same way
    X_test = np.random.randn(n_test, n_features)
    y_test = (X_test[:, 0] + X_test[:, 1] + np.random.randn(n_test) * 0.5 > 0).astype(int)

    # Return the four arrays
    return X_train, X_test, y_train, y_test


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test outputs.
    Cleans up after the test.
    """
    # Create a temporary directory
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Cleanup: remove the temporary directory
    shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Tests for ModelTrainer
# ============================================================

class TestModelTrainer:
    """Test suite for the ModelTrainer class."""

    def test_init(self, config):
        """Test that ModelTrainer initializes correctly."""
        # Create a trainer instance
        trainer = ModelTrainer(config)
        # Verify it was created
        assert trainer is not None

    def test_train_all_returns_dict(self, config, binary_dataset):
        """Test that train_all_models returns a dictionary of models."""
        # Unpack the dataset
        X_train, X_test, y_train, y_test = binary_dataset
        # Create trainer and train
        trainer = ModelTrainer(config)
        models = trainer.train_all_models(X_train, y_train)
        # Should return a dictionary
        assert isinstance(models, dict), f"Expected dict, got {type(models)}"
        # Should have at least one model
        assert len(models) > 0, "No models were trained"

    def test_trained_models_have_predict_proba(self, config, binary_dataset):
        """Test that all trained models support predict_proba."""
        # Unpack the dataset
        X_train, X_test, y_train, y_test = binary_dataset
        # Train models
        trainer = ModelTrainer(config)
        models = trainer.train_all_models(X_train, y_train)
        # Check each model
        for name, model in models.items():
            # Should have predict_proba method
            assert hasattr(model, "predict_proba"), (
                f"Model '{name}' missing predict_proba method"
            )
            # Predictions should have correct shape
            proba = model.predict_proba(X_test)
            assert proba.shape == (len(X_test), 2), (
                f"Model '{name}' predict_proba shape is {proba.shape}, "
                f"expected ({len(X_test)}, 2)"
            )

    def test_probabilities_in_valid_range(self, config, binary_dataset):
        """Test that predicted probabilities are between 0 and 1."""
        # Unpack the dataset
        X_train, X_test, y_train, y_test = binary_dataset
        # Train models
        trainer = ModelTrainer(config)
        models = trainer.train_all_models(X_train, y_train)
        # Check each model's probabilities
        for name, model in models.items():
            proba = model.predict_proba(X_test)[:, 1]
            # All probabilities should be in [0, 1]
            assert (proba >= 0).all(), f"Model '{name}' has negative probabilities"
            assert (proba <= 1).all(), f"Model '{name}' has probabilities > 1"

    def test_find_optimal_threshold(self, config, binary_dataset):
        """Test that threshold optimization returns a valid threshold."""
        # Unpack the dataset
        X_train, X_test, y_train, y_test = binary_dataset
        # Train a single model
        trainer = ModelTrainer(config)
        models = trainer.train_all_models(X_train, y_train)
        # Get the first model
        model_name = list(models.keys())[0]
        model = models[model_name]
        # Find optimal threshold
        threshold = trainer.find_optimal_threshold(model, X_test, y_test)
        # Threshold should be between 0 and 1
        assert 0 < threshold < 1, f"Threshold {threshold} is out of range (0, 1)"


# ============================================================
# Tests for ModelEvaluator
# ============================================================

class TestModelEvaluator:
    """Test suite for the ModelEvaluator class."""

    def test_init(self, config):
        """Test that ModelEvaluator initializes correctly."""
        # Create an evaluator instance
        evaluator = ModelEvaluator(config)
        # Verify creation
        assert evaluator is not None

    def test_compute_metrics_returns_dict(self, config, binary_dataset):
        """Test that compute_metrics returns a dictionary with expected keys."""
        # Unpack and train
        X_train, X_test, y_train, y_test = binary_dataset
        trainer = ModelTrainer(config)
        models = trainer.train_all_models(X_train, y_train)
        model = list(models.values())[0]
        # Evaluate
        evaluator = ModelEvaluator(config)
        metrics = evaluator.compute_metrics(model, X_test, y_test)
        # Should return a dictionary
        assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
        # Should contain key metrics
        expected_keys = ["roc_auc", "f1"]
        for key in expected_keys:
            assert key in metrics, f"Missing expected metric: {key}"

    def test_roc_auc_in_valid_range(self, config, binary_dataset):
        """Test that ROC-AUC is between 0 and 1."""
        # Unpack and train
        X_train, X_test, y_train, y_test = binary_dataset
        trainer = ModelTrainer(config)
        models = trainer.train_all_models(X_train, y_train)
        # Evaluate each model
        evaluator = ModelEvaluator(config)
        for name, model in models.items():
            metrics = evaluator.compute_metrics(model, X_test, y_test)
            roc_auc = metrics.get("roc_auc", 0)
            assert 0 <= roc_auc <= 1, f"Model '{name}' ROC-AUC {roc_auc} out of range"

    def test_metrics_above_random(self, config, binary_dataset):
        """Test that trained model performs above random chance (ROC-AUC > 0.5)."""
        # Unpack and train
        X_train, X_test, y_train, y_test = binary_dataset
        trainer = ModelTrainer(config)
        models = trainer.train_all_models(X_train, y_train)
        # Evaluate
        evaluator = ModelEvaluator(config)
        for name, model in models.items():
            metrics = evaluator.compute_metrics(model, X_test, y_test)
            roc_auc = metrics.get("roc_auc", 0)
            # Model should beat random baseline
            assert roc_auc > 0.5, (
                f"Model '{name}' ROC-AUC {roc_auc:.3f} is below random chance"
            )

    def test_compute_lift_at_k(self, config, binary_dataset):
        """Test that lift@k computation returns a valid value."""
        # Unpack and train
        X_train, X_test, y_train, y_test = binary_dataset
        trainer = ModelTrainer(config)
        models = trainer.train_all_models(X_train, y_train)
        model = list(models.values())[0]
        # Compute lift at 10%
        evaluator = ModelEvaluator(config)
        lift = evaluator.compute_lift_at_k(model, X_test, y_test, k=0.1)
        # Lift should be a positive number
        assert lift > 0, f"Lift@10% should be positive, got {lift}"


# ============================================================
# Tests for ModelRegistry
# ============================================================

class TestModelRegistry:
    """Test suite for the ModelRegistry class."""

    def test_init(self, config, temp_dir):
        """Test that ModelRegistry initializes and creates directory."""
        # Override registry dir to use temp directory
        custom_config = config.copy()
        custom_config["model_registry"] = {"dir": temp_dir}
        # Create registry
        registry = ModelRegistry(custom_config)
        # Verify it was created
        assert registry is not None

    def test_register_and_list(self, config, temp_dir, binary_dataset):
        """Test model registration and listing."""
        # Setup
        custom_config = config.copy()
        custom_config["model_registry"] = {"dir": temp_dir}
        X_train, X_test, y_train, y_test = binary_dataset
        # Train a model
        trainer = ModelTrainer(config)
        models = trainer.train_all_models(X_train, y_train)
        model_name = list(models.keys())[0]
        model = models[model_name]
        # Register the model
        registry = ModelRegistry(custom_config)
        version = registry.register_model(
            model=model,
            model_name=model_name,
            metrics={"roc_auc": 0.85},
            config=config,
        )
        # Version should be a positive integer
        assert version >= 1, f"Model version should be >= 1, got {version}"
        # List models
        model_list = registry.list_models()
        # Should have at least one model
        assert len(model_list) > 0, "No models found after registration"

    def test_register_increments_version(self, config, temp_dir, binary_dataset):
        """Test that registering the same model increments the version."""
        # Setup
        custom_config = config.copy()
        custom_config["model_registry"] = {"dir": temp_dir}
        X_train, _, y_train, _ = binary_dataset
        # Train a model
        trainer = ModelTrainer(config)
        models = trainer.train_all_models(X_train, y_train)
        model_name = list(models.keys())[0]
        model = models[model_name]
        # Register the same model twice
        registry = ModelRegistry(custom_config)
        v1 = registry.register_model(model=model, model_name=model_name,
                                      metrics={"roc_auc": 0.80}, config=config)
        v2 = registry.register_model(model=model, model_name=model_name,
                                      metrics={"roc_auc": 0.85}, config=config)
        # Second version should be higher
        assert v2 > v1, f"Version did not increment: v1={v1}, v2={v2}"
