# ============================================================
# src/actions/recommender.py
# Next-Best-Action recommender: suggests personalized retention
# actions for at-risk customers based on their churn drivers.
# ============================================================

import pandas as pd                                # Pandas for DataFrames
import numpy as np                                 # NumPy for numerical operations
from loguru import logger                          # Structured logging
from typing import Dict, Any, List, Optional       # Type hints


class ActionRecommender:
    """
    Recommends personalized retention actions for at-risk customers.

    Maps churn drivers (from SHAP explanations) to specific,
    actionable retention strategies. Prioritizes actions by
    expected ROI (impact / cost).

    Action categories:
    1. Pricing actions (discounts, plan changes)
    2. Service upgrades (add security, entertainment)
    3. Contract incentives (lock-in offers)
    4. Engagement actions (outreach, loyalty programs)
    """

    def __init__(self, config: dict):
        """
        Initialize the ActionRecommender with business rules.

        Parameters
        ----------
        config : dict
            Project configuration (for cost/value parameters).
        """
        # Store the configuration
        self.config = config
        # Extract retention cost from config
        self.retention_cost = config["model"]["threshold"]["retention_cost"]
        # Extract average customer value from config
        self.customer_value = config["model"]["threshold"]["avg_customer_value"]

        # Define the action catalog: maps churn drivers to recommended actions
        # Each action has a name, description, estimated cost, and expected impact
        self.action_catalog = self._build_action_catalog()

        # Log initialization
        logger.info("ActionRecommender initialized with action catalog")

    def _build_action_catalog(self) -> Dict[str, Dict]:
        """
        Build the catalog of possible retention actions.

        Each action is mapped to the churn driver it addresses.

        Returns
        -------
        dict
            Dictionary mapping driver keywords to action details.
        """
        # Define all available retention actions
        catalog = {
            # --- Pricing Actions ---
            "MonthlyCharges": {
                "action": "Offer promotional discount",
                "description": "Offer a 15-20% discount on monthly charges for 6 months",
                "category": "pricing",
                "estimated_cost": 60,              # Cost per customer over 6 months
                "expected_retention_lift": 0.15,   # 15% improvement in retention probability
                "priority": "high",
            },
            "charge_increase": {
                "action": "Price match or rollback",
                "description": "Revert recent price increase or match competitor pricing",
                "category": "pricing",
                "estimated_cost": 40,
                "expected_retention_lift": 0.20,
                "priority": "high",
            },
            "charge_per_service": {
                "action": "Bundle discount",
                "description": "Offer a bundled service plan at a reduced per-service rate",
                "category": "pricing",
                "estimated_cost": 30,
                "expected_retention_lift": 0.10,
                "priority": "medium",
            },

            # --- Contract Actions ---
            "is_month_to_month": {
                "action": "Offer annual contract incentive",
                "description": "Offer a significant discount (20-30%) for switching to yearly contract",
                "category": "contract",
                "estimated_cost": 80,
                "expected_retention_lift": 0.35,
                "priority": "high",
            },
            "Contract": {
                "action": "Contract upgrade incentive",
                "description": "Offer benefits for committing to a longer contract term",
                "category": "contract",
                "estimated_cost": 50,
                "expected_retention_lift": 0.25,
                "priority": "high",
            },

            # --- Service Actions ---
            "no_protection": {
                "action": "Free security trial",
                "description": "Offer 3-month free trial of Online Security + Device Protection",
                "category": "service",
                "estimated_cost": 45,
                "expected_retention_lift": 0.20,
                "priority": "high",
            },
            "security_score": {
                "action": "Security bundle upgrade",
                "description": "Upgrade security package with discounted rate",
                "category": "service",
                "estimated_cost": 35,
                "expected_retention_lift": 0.15,
                "priority": "medium",
            },
            "is_fiber_optic": {
                "action": "Fiber experience improvement",
                "description": "Provide speed upgrade or equipment replacement for fiber customers",
                "category": "service",
                "estimated_cost": 25,
                "expected_retention_lift": 0.10,
                "priority": "medium",
            },
            "total_services": {
                "action": "Cross-sell additional services",
                "description": "Offer free trial of streaming or tech support services",
                "category": "service",
                "estimated_cost": 20,
                "expected_retention_lift": 0.12,
                "priority": "medium",
            },

            # --- Payment Actions ---
            "uses_electronic_check": {
                "action": "Switch to auto-pay incentive",
                "description": "Offer a $5/month discount for switching to automatic payment",
                "category": "payment",
                "estimated_cost": 60,
                "expected_retention_lift": 0.18,
                "priority": "medium",
            },
            "paperless_echeck_risk": {
                "action": "Payment method optimization",
                "description": "Offer incentive to switch to auto-pay + paper billing option",
                "category": "payment",
                "estimated_cost": 40,
                "expected_retention_lift": 0.15,
                "priority": "medium",
            },

            # --- Engagement Actions ---
            "tenure": {
                "action": "Loyalty reward program",
                "description": "Offer loyalty points, anniversary gifts, or tenure-based discounts",
                "category": "engagement",
                "estimated_cost": 25,
                "expected_retention_lift": 0.10,
                "priority": "low",
            },
            "is_new_customer": {
                "action": "Enhanced onboarding program",
                "description": "Assign a dedicated success manager for the first 90 days",
                "category": "engagement",
                "estimated_cost": 30,
                "expected_retention_lift": 0.20,
                "priority": "high",
            },
            "SeniorCitizen": {
                "action": "Senior-specific support plan",
                "description": "Offer dedicated senior support line and simplified billing",
                "category": "engagement",
                "estimated_cost": 15,
                "expected_retention_lift": 0.08,
                "priority": "low",
            },
        }

        # Log the catalog size
        logger.info(f"Action catalog built: {len(catalog)} actions available")

        # Return the catalog
        return catalog

    def recommend_actions(
        self,
        customer_explanation: Dict[str, Any],
        max_actions: int = 3,
        budget_limit: float = 200.0,
    ) -> List[Dict[str, Any]]:
        """
        Recommend personalized retention actions for a customer.

        Uses the SHAP-based explanation to identify the top churn
        drivers, then maps each driver to an actionable intervention.

        Parameters
        ----------
        customer_explanation : dict
            Output from ModelExplainer.explain_customer().
            Must contain 'churn_probability' and 'top_churn_drivers'.
        max_actions : int
            Maximum number of actions to recommend.
        budget_limit : float
            Maximum total cost of all recommended actions.

        Returns
        -------
        list of dict
            Ordered list of recommended actions with expected impact.
        """
        # Extract churn probability and risk level
        churn_prob = customer_explanation["churn_probability"]
        risk_level = customer_explanation["risk_level"]

        # Log the recommendation process
        logger.info(f"Generating recommendations for customer (risk={risk_level}, P={churn_prob:.1%})")

        # Get the top churn drivers from the explanation
        drivers = customer_explanation.get("top_churn_drivers", [])

        # Match drivers to actions from the catalog
        matched_actions = []
        for driver in drivers:
            feature = driver["feature"]
            impact = driver["impact"]

            # Check if this feature maps to an action in the catalog
            # Try exact match first, then partial match
            matched_action = None
            for key in self.action_catalog:
                # Check if the driver feature contains the catalog key
                if key in feature or feature in key:
                    matched_action = self.action_catalog[key].copy()
                    matched_action["triggered_by"] = feature
                    matched_action["driver_impact"] = round(float(impact), 4)
                    break

            # If we found a matching action, add it to the list
            if matched_action:
                matched_actions.append(matched_action)

        # If no specific actions matched, add a general retention action
        if not matched_actions:
            matched_actions.append({
                "action": "Proactive retention outreach",
                "description": "Contact customer to understand pain points and offer personalized solution",
                "category": "engagement",
                "estimated_cost": 20,
                "expected_retention_lift": 0.10,
                "priority": "medium",
                "triggered_by": "general_risk",
                "driver_impact": 0,
            })

        # Calculate expected ROI for each action
        for action in matched_actions:
            # Expected value saved = churn_prob × retention_lift × customer_value
            expected_save = (
                churn_prob *                           # Current churn probability
                action["expected_retention_lift"] *    # Expected lift from action
                self.customer_value                    # Customer's annual value
            )
            # ROI = (expected savings - cost) / cost
            cost = action["estimated_cost"]
            roi = (expected_save - cost) / cost if cost > 0 else 0
            # Add ROI and expected savings to the action
            action["expected_savings"] = round(expected_save, 2)
            action["roi"] = round(roi, 2)

        # Sort actions by ROI (highest first)
        matched_actions.sort(key=lambda x: x["roi"], reverse=True)

        # Apply budget constraint: select actions within budget
        selected_actions = []
        remaining_budget = budget_limit
        for action in matched_actions:
            # Check if this action fits within the remaining budget
            if action["estimated_cost"] <= remaining_budget and len(selected_actions) < max_actions:
                # Add the action to the selection
                selected_actions.append(action)
                # Deduct the cost from the remaining budget
                remaining_budget -= action["estimated_cost"]

        # Log the recommendations
        logger.info(f"Recommended {len(selected_actions)} actions:")
        for i, action in enumerate(selected_actions, 1):
            logger.info(f"  {i}. {action['action']} (ROI={action['roi']:.1f}x, Cost=${action['estimated_cost']})")

        # Return the selected actions
        return selected_actions

    def generate_retention_plan(
        self,
        customer_explanation: Dict[str, Any],
        customer_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive retention plan for a customer.

        This combines the risk assessment, explanations, recommended
        actions, and expected outcomes into a single actionable plan.

        Parameters
        ----------
        customer_explanation : dict
            Output from ModelExplainer.explain_customer().
        customer_data : dict, optional
            Additional customer metadata (name, account, etc.).

        Returns
        -------
        dict
            Complete retention plan with all details.
        """
        # Get the churn probability and risk level
        churn_prob = customer_explanation["churn_probability"]
        risk_level = customer_explanation["risk_level"]

        # Generate action recommendations
        recommended_actions = self.recommend_actions(customer_explanation)

        # Calculate total expected impact
        total_cost = sum(a["estimated_cost"] for a in recommended_actions)
        total_expected_savings = sum(a["expected_savings"] for a in recommended_actions)
        net_value = total_expected_savings - total_cost

        # Determine urgency based on risk level
        urgency_map = {
            "HIGH": "IMMEDIATE",                       # Act within 24 hours
            "MEDIUM": "THIS_WEEK",                     # Act within the week
            "LOW": "NEXT_REVIEW",                      # Review at next cycle
        }
        urgency = urgency_map.get(risk_level, "NEXT_REVIEW")

        # Build the retention plan
        plan = {
            "risk_assessment": {
                "churn_probability": churn_prob,
                "risk_level": risk_level,
                "urgency": urgency,
                "expected_loss_if_churned": round(self.customer_value * churn_prob, 2),
            },
            "churn_drivers": customer_explanation.get("top_churn_drivers", []),
            "recommended_actions": recommended_actions,
            "financial_summary": {
                "total_retention_cost": total_cost,
                "total_expected_savings": round(total_expected_savings, 2),
                "net_expected_value": round(net_value, 2),
                "action_recommended": net_value > 0,
            },
            "customer_info": customer_data or {},
        }

        # Log the plan summary
        logger.info(
            f"Retention plan generated: "
            f"Risk={risk_level}, Urgency={urgency}, "
            f"Net value=${net_value:.0f}"
        )

        # Return the complete retention plan
        return plan

    def prioritize_customers(
        self,
        customers: pd.DataFrame,
        churn_probabilities: np.ndarray,
    ) -> pd.DataFrame:
        """
        Prioritize a batch of customers for retention outreach.

        Ranks customers by expected loss (churn_prob × customer_value),
        not just churn probability, because losing a high-value customer
        matters more than losing a low-value one.

        Parameters
        ----------
        customers : pd.DataFrame
            Customer feature DataFrame.
        churn_probabilities : np.ndarray
            Predicted churn probabilities for each customer.

        Returns
        -------
        pd.DataFrame
            Prioritized customer DataFrame with risk scores and rankings.
        """
        # Create a copy to avoid modifying the original
        result = customers.copy()

        # Add churn probability column
        result["churn_probability"] = churn_probabilities

        # Calculate expected loss for each customer
        # If MonthlyCharges is available, use it as a value proxy
        if "MonthlyCharges" in result.columns:
            # Annualized customer value = MonthlyCharges × 12
            result["annual_value"] = result["MonthlyCharges"] * 12
        else:
            # Use default customer value from config
            result["annual_value"] = self.customer_value

        # Expected loss = churn probability × customer value
        result["expected_loss"] = result["churn_probability"] * result["annual_value"]

        # Assign risk level based on churn probability thresholds
        result["risk_level"] = pd.cut(
            result["churn_probability"],
            bins=[0, 0.3, 0.6, 1.0],                 # Risk tier boundaries
            labels=["LOW", "MEDIUM", "HIGH"],          # Risk labels
            include_lowest=True,                       # Include 0 in the first bin
        )

        # Sort by expected loss (highest first = most important to retain)
        result = result.sort_values("expected_loss", ascending=False)

        # Add priority ranking
        result["priority_rank"] = range(1, len(result) + 1)

        # Reset index for clean display
        result = result.reset_index(drop=True)

        # Log the prioritization summary
        high_risk = (result["risk_level"] == "HIGH").sum()
        medium_risk = (result["risk_level"] == "MEDIUM").sum()
        logger.info(
            f"Customer prioritization complete: "
            f"{high_risk} HIGH risk, {medium_risk} MEDIUM risk, "
            f"Total expected loss: ${result['expected_loss'].sum():,.0f}"
        )

        # Return the prioritized DataFrame
        return result
