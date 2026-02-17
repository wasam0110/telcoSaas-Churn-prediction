# ============================================================
# api/schemas.py
# Pydantic models for API request/response validation.
# Defines the exact shape of data flowing in and out of the API.
# ============================================================

from pydantic import BaseModel, Field              # Pydantic for data validation
from typing import List, Dict, Any, Optional       # Type hints
from enum import Enum                              # Enumerations for constrained fields


class RiskLevel(str, Enum):
    """
    Enumeration of possible churn risk levels.
    Constrains the risk_level field to valid values only.
    """
    LOW = "LOW"                                    # Churn probability < 40%
    MEDIUM = "MEDIUM"                              # Churn probability 40-70%
    HIGH = "HIGH"                                  # Churn probability >= 70%


class CustomerFeatures(BaseModel):
    """
    Input schema for a single customer's features.

    All fields correspond to the Telco Customer Churn dataset columns.
    Each field has a description and constraints for validation.
    """
    # --- Demographic Features ---
    gender: str = Field(
        ...,                                       # Required field (no default)
        description="Customer gender: 'Male' or 'Female'",
        examples=["Male"],
    )
    SeniorCitizen: int = Field(
        ...,
        description="Whether the customer is a senior citizen: 0 or 1",
        ge=0,                                      # Greater than or equal to 0
        le=1,                                      # Less than or equal to 1
        examples=[0],
    )
    Partner: str = Field(
        ...,
        description="Whether the customer has a partner: 'Yes' or 'No'",
        examples=["Yes"],
    )
    Dependents: str = Field(
        ...,
        description="Whether the customer has dependents: 'Yes' or 'No'",
        examples=["No"],
    )

    # --- Account Features ---
    tenure: int = Field(
        ...,
        description="Number of months the customer has been with the company",
        ge=0,                                      # Cannot be negative
        examples=[12],
    )
    Contract: str = Field(
        ...,
        description="Contract type: 'Month-to-month', 'One year', or 'Two year'",
        examples=["Month-to-month"],
    )
    PaperlessBilling: str = Field(
        ...,
        description="Whether the customer uses paperless billing: 'Yes' or 'No'",
        examples=["Yes"],
    )
    PaymentMethod: str = Field(
        ...,
        description="Payment method used by the customer",
        examples=["Electronic check"],
    )
    MonthlyCharges: float = Field(
        ...,
        description="Monthly charge amount in dollars",
        ge=0,                                      # Cannot be negative
        examples=[70.35],
    )
    TotalCharges: float = Field(
        ...,
        description="Total charges accumulated over the customer's tenure",
        ge=0,                                      # Cannot be negative
        examples=[844.2],
    )

    # --- Phone Services ---
    PhoneService: str = Field(
        ...,
        description="Whether the customer has phone service: 'Yes' or 'No'",
        examples=["Yes"],
    )
    MultipleLines: str = Field(
        ...,
        description="Whether the customer has multiple lines: 'Yes', 'No', or 'No phone service'",
        examples=["No"],
    )

    # --- Internet Services ---
    InternetService: str = Field(
        ...,
        description="Type of internet service: 'DSL', 'Fiber optic', or 'No'",
        examples=["Fiber optic"],
    )
    OnlineSecurity: str = Field(
        ...,
        description="Whether the customer has online security: 'Yes', 'No', or 'No internet service'",
        examples=["No"],
    )
    OnlineBackup: str = Field(
        ...,
        description="Whether the customer has online backup: 'Yes', 'No', or 'No internet service'",
        examples=["No"],
    )
    DeviceProtection: str = Field(
        ...,
        description="Whether the customer has device protection: 'Yes', 'No', or 'No internet service'",
        examples=["No"],
    )
    TechSupport: str = Field(
        ...,
        description="Whether the customer has tech support: 'Yes', 'No', or 'No internet service'",
        examples=["No"],
    )
    StreamingTV: str = Field(
        ...,
        description="Whether the customer has streaming TV: 'Yes', 'No', or 'No internet service'",
        examples=["No"],
    )
    StreamingMovies: str = Field(
        ...,
        description="Whether the customer has streaming movies: 'Yes', 'No', or 'No internet service'",
        examples=["No"],
    )


class PredictionResponse(BaseModel):
    """
    Output schema for a churn prediction response.

    Contains the churn probability, risk level, top churn drivers,
    and recommended retention actions.
    """
    churn_probability: float = Field(
        ...,
        description="Predicted probability of churn (0.0 to 1.0)",
        ge=0.0,                                    # Minimum probability
        le=1.0,                                    # Maximum probability
        examples=[0.73],
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Categorized risk level based on churn probability",
        examples=["HIGH"],
    )
    threshold_used: float = Field(
        ...,
        description="Decision threshold used for classification",
        examples=[0.45],
    )
    will_churn: bool = Field(
        ...,
        description="Binary prediction: True if churn probability exceeds threshold",
        examples=[True],
    )
    top_drivers: List[Dict[str, Any]] = Field(
        default=[],
        description="Top features driving this customer's churn risk",
    )
    recommended_actions: List[Dict[str, Any]] = Field(
        default=[],
        description="Recommended retention actions for this customer",
    )


class BatchPredictionRequest(BaseModel):
    """
    Input schema for batch prediction (multiple customers at once).
    """
    customers: List[CustomerFeatures] = Field(
        ...,
        description="List of customer feature sets for batch prediction",
        min_length=1,                              # At least one customer required
        max_length=1000,                           # Cap at 1000 for performance
    )


class BatchPredictionResponse(BaseModel):
    """
    Output schema for batch prediction results.
    """
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of prediction results for each customer",
    )
    summary: Dict[str, Any] = Field(
        default={},
        description="Summary statistics for the batch (e.g., average risk, distribution)",
    )


class HealthResponse(BaseModel):
    """
    Output schema for the health check endpoint.
    """
    status: str = Field(
        ...,
        description="API health status",
        examples=["healthy"],
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the prediction model is loaded and ready",
        examples=[True],
    )
    version: str = Field(
        ...,
        description="API version string",
        examples=["1.0.0"],
    )


class WhatIfRequest(BaseModel):
    """
    Input schema for what-if analysis.

    Allows the user to propose feature changes and see
    how the churn prediction would change.
    """
    customer: CustomerFeatures = Field(
        ...,
        description="Current customer features",
    )
    changes: Dict[str, Any] = Field(
        ...,
        description="Dictionary of feature_name: new_value pairs to simulate",
        examples=[{"Contract": "One year", "MonthlyCharges": 60.0}],
    )


class WhatIfResponse(BaseModel):
    """
    Output schema for what-if analysis results.
    """
    original_probability: float = Field(
        ...,
        description="Churn probability before the proposed changes",
    )
    new_probability: float = Field(
        ...,
        description="Churn probability after the proposed changes",
    )
    probability_change: float = Field(
        ...,
        description="Absolute change in churn probability",
    )
    risk_reduced: bool = Field(
        ...,
        description="Whether the proposed changes reduce churn risk",
    )
    changes_applied: Dict[str, Any] = Field(
        ...,
        description="The feature changes that were applied",
    )
