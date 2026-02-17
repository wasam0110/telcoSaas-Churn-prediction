# ============================================================
# src/data/loader.py
# Responsible for loading raw data from various sources
# (CSV, database, API) with schema validation.
# ============================================================

import pandas as pd                                # Pandas for tabular data operations
from pathlib import Path                           # Object-oriented file paths
from loguru import logger                          # Structured logging
from typing import Optional, List                  # Type hints for function signatures


class DataLoader:
    """
    Loads raw telco customer data from CSV files.

    This class handles reading data, initial type casting,
    and basic sanity checks before passing data downstream.
    """

    def __init__(self, config: dict):
        """
        Initialize the DataLoader with project configuration.

        Parameters
        ----------
        config : dict
            The parsed config.yaml dictionary containing data paths and settings.
        """
        # Store the full configuration for access to all settings
        self.config = config
        # Extract the raw data file path from config
        self.raw_path = config["data"]["raw_path"]
        # Extract the target column name (e.g., "Churn")
        self.target = config["features"]["target"]
        # Log that the DataLoader has been initialized
        logger.info(f"DataLoader initialized. Raw data path: {self.raw_path}")

    def load_csv(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame with initial cleaning.

        Parameters
        ----------
        filepath : str, optional
            Override path to the CSV file. If None, uses the config path.

        Returns
        -------
        pd.DataFrame
            The loaded and initially cleaned DataFrame.
        """
        # Use the provided filepath or fall back to the config default
        path = Path(filepath) if filepath else Path(self.raw_path)

        # Verify the file exists before attempting to read
        if not path.exists():
            # Raise an error with the exact path for debugging
            raise FileNotFoundError(f"Data file not found: {path}")

        # Log the start of data loading
        logger.info(f"Loading data from: {path}")

        # Read the CSV file into a DataFrame
        # low_memory=False ensures consistent dtype inference across chunks
        df = pd.read_csv(path, low_memory=False)

        # Log the shape of the loaded data for verification
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Perform initial cleaning: strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Log all column names for debugging
        logger.debug(f"Columns: {list(df.columns)}")

        # Return the loaded DataFrame
        return df

    def _fix_total_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix the TotalCharges column which may have blank strings
        instead of NaN for new customers with zero tenure.

        Parameters
        ----------
        df : pd.DataFrame
            The raw DataFrame with potential TotalCharges issues.

        Returns
        -------
        pd.DataFrame
            DataFrame with TotalCharges converted to numeric.
        """
        # Check if TotalCharges exists in the DataFrame
        if "TotalCharges" in df.columns:
            # Replace empty strings and whitespace-only values with NaN
            df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
            # Convert TotalCharges to numeric, coercing errors to NaN
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            # Count how many NaN values resulted from conversion
            null_count = df["TotalCharges"].isna().sum()
            # Log the count of converted null values
            logger.info(f"TotalCharges: {null_count} blank values converted to NaN")

        # Return the DataFrame with fixed TotalCharges
        return df

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the target column from 'Yes'/'No' strings to binary 1/0.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a string-valued target column.

        Returns
        -------
        pd.DataFrame
            DataFrame with the target column encoded as 0/1 integers.
        """
        # Check if the target column exists
        if self.target in df.columns:
            # Map 'Yes' to 1 (churned) and 'No' to 0 (retained)
            df[self.target] = df[self.target].map({"Yes": 1, "No": 0})
            # Log the class distribution for imbalance awareness
            churn_rate = df[self.target].mean()
            logger.info(f"Target encoded. Churn rate: {churn_rate:.2%}")

        # Return the DataFrame with encoded target
        return df

    def load_and_prepare(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Complete data loading pipeline: load CSV, fix types, encode target.

        This is the main entry point for getting data ready for preprocessing.

        Parameters
        ----------
        filepath : str, optional
            Override path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame ready for preprocessing and feature engineering.
        """
        # Step 1: Load the raw CSV file
        df = self.load_csv(filepath)

        # Step 2: Fix the TotalCharges column (blank strings -> NaN)
        df = self._fix_total_charges(df)

        # Step 3: Encode the target variable (Yes/No -> 1/0)
        df = self._encode_target(df)

        # Step 4: Drop the customerID column if present (not a feature)
        if "customerID" in df.columns:
            # Save customerIDs separately in case we need them later
            self.customer_ids = df["customerID"].copy()
            # Drop customerID from the feature DataFrame
            df = df.drop(columns=["customerID"])
            # Log that we removed the ID column
            logger.info("Dropped 'customerID' column (not a predictive feature)")

        # Log final shape after initial preparation
        logger.info(f"Data preparation complete. Final shape: {df.shape}")

        # Return the prepared DataFrame
        return df
