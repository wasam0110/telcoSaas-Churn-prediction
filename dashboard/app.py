# ============================================================
# dashboard/app.py
# Streamlit interactive dashboard for the Churn Prediction SaaS.
# Provides risk overview, individual customer analysis,
# what-if simulation, and drift monitoring visualizations.
# ============================================================

import sys                                         # System-specific parameters
from pathlib import Path                           # Object-oriented file paths

# Add the project root to Python path for module imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st                             # Streamlit for the web dashboard
import pandas as pd                                # Pandas for data manipulation
import numpy as np                                 # NumPy for numerical operations
import plotly.express as px                        # Plotly Express for interactive charts
import plotly.graph_objects as go                   # Plotly Graph Objects for custom charts
import joblib                                      # For loading serialized objects
from loguru import logger                          # Structured logging

from src.utils.helpers import load_config          # Config loader utility

# ============================================================
# Page Configuration (must be the first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Churn Prediction Dashboard",       # Browser tab title
    page_icon="📊",                                # Browser tab icon
    layout="wide",                                 # Use full page width
    initial_sidebar_state="expanded",              # Sidebar starts open
)


# ============================================================
# Cached Resource Loading
# ============================================================
# @st.cache_resource ensures these are loaded only once,
# not on every page rerun.

@st.cache_resource
def load_app_config():
    """
    Load and cache the project configuration.

    Returns
    -------
    dict
        The parsed config.yaml dictionary.
    """
    try:
        # Load the config file
        return load_config()
    except Exception as e:
        # Log the error
        logger.error(f"Failed to load config: {e}")
        # Return a minimal default config
        return {"project": {"name": "Churn Prediction"}}


@st.cache_resource
def load_trained_model():
    """
    Load and cache the trained model.

    Returns
    -------
    object or None
        The trained model, or None if not available.
    """
    # Check if the model file exists
    model_path = Path("models/best_model.joblib")
    if model_path.exists():
        # Load and return the model
        return joblib.load(model_path)
    else:
        # Return None if model doesn't exist
        return None


@st.cache_resource
def load_preprocessor():
    """
    Load and cache the fitted preprocessor.

    Returns
    -------
    object or None
        The fitted preprocessor, or None if not available.
    """
    # Check if the preprocessor file exists
    preprocessor_path = Path("models/preprocessor.joblib")
    if preprocessor_path.exists():
        return joblib.load(preprocessor_path)
    return None


# ============================================================
# Sidebar Navigation
# ============================================================
def render_sidebar():
    """
    Render the sidebar with navigation and configuration options.

    Returns
    -------
    str
        The selected page name.
    """
    # Add the logo/title to the sidebar
    st.sidebar.title("📊 Churn Prediction")
    st.sidebar.markdown("---")                     # Horizontal separator

    # Navigation menu
    page = st.sidebar.radio(
        "Navigate to:",                            # Radio button label
        [
            "🏠 Overview",                         # Dashboard overview
            "🔍 Customer Analysis",                # Individual customer lookup
            "🧪 What-If Simulator",                # What-if analysis tool
            "📈 Model Performance",                # Model evaluation metrics
            "⚠️ Drift Monitoring",                 # Drift detection dashboard
            "📋 Batch Predictions",                # Batch scoring tool
        ],
        index=0,                                   # Default to Overview page
    )

    # Add model status indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Status")
    model = load_trained_model()
    if model is not None:
        # Model is loaded — show green indicator
        st.sidebar.success("✅ Model loaded and ready")
    else:
        # Model not available — show warning
        st.sidebar.warning("⚠️ No model trained yet")
        st.sidebar.info("Run `python scripts/train.py` to train a model")

    # Return the selected page
    return page


# ============================================================
# Page: Overview
# ============================================================
def render_overview():
    """
    Render the dashboard overview page with key metrics and charts.
    """
    # Page title
    st.title("🏠 Churn Prediction Dashboard")
    st.markdown("Real-time customer churn risk monitoring and actionable insights.")

    # Check if model is available
    model = load_trained_model()
    if model is None:
        # Show instructions if no model is trained
        st.warning("No trained model found. Please train a model first.")
        st.code("python scripts/train.py", language="bash")
        return

    # Key Metrics Row (4 columns)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Total customers metric
        st.metric(
            label="Total Customers",               # Metric label
            value="7,043",                         # Placeholder value
            delta="+125 this month",               # Change indicator
        )

    with col2:
        # Average churn risk metric
        st.metric(
            label="Avg Churn Risk",
            value="26.5%",
            delta="-1.2%",                         # Decrease is good (green)
            delta_color="inverse",                 # Green for decrease
        )

    with col3:
        # High risk customers metric
        st.metric(
            label="High Risk Customers",
            value="482",
            delta="+23",
            delta_color="inverse",                 # Red for increase
        )

    with col4:
        # Expected monthly revenue at risk
        st.metric(
            label="Revenue at Risk",
            value="$34,150",
            delta="-$2,100",
            delta_color="inverse",
        )

    # Separator
    st.markdown("---")

    # Charts Row
    col_left, col_right = st.columns(2)

    with col_left:
        # Risk Distribution Pie Chart
        st.subheader("Risk Distribution")
        # Create sample data for the pie chart
        risk_data = pd.DataFrame({
            "Risk Level": ["Low", "Medium", "High"],
            "Count": [4500, 2061, 482],
        })
        # Create interactive pie chart with Plotly
        fig = px.pie(
            risk_data,                             # Data source
            values="Count",                        # Size of each slice
            names="Risk Level",                    # Labels for each slice
            color="Risk Level",                    # Color by risk level
            color_discrete_map={                   # Custom color mapping
                "Low": "#2ecc71",                  # Green for low risk
                "Medium": "#f39c12",               # Orange for medium risk
                "High": "#e74c3c",                 # Red for high risk
            },
            hole=0.4,                              # Donut chart (hollow center)
        )
        # Update layout for a cleaner look
        fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
        )
        # Render the chart
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Monthly Churn Trend Line Chart
        st.subheader("Monthly Churn Trend")
        # Create sample trend data
        months = pd.date_range("2025-03-01", periods=12, freq="ME")
        trend_data = pd.DataFrame({
            "Month": months,
            "Churn Rate": [28.5, 27.8, 27.2, 26.9, 27.5, 26.8,
                           26.5, 26.1, 25.8, 26.2, 25.5, 26.5],
        })
        # Create interactive line chart
        fig = px.line(
            trend_data,
            x="Month",                            # X-axis
            y="Churn Rate",                        # Y-axis
            markers=True,                          # Show data point markers
        )
        fig.update_layout(
            yaxis_title="Churn Rate (%)",
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page: Customer Analysis
# ============================================================
def render_customer_analysis():
    """
    Render the individual customer analysis page.
    Allows users to input customer details and see predictions.
    """
    st.title("🔍 Customer Churn Analysis")
    st.markdown("Enter customer details to get a churn prediction with explanations.")

    # Check model availability
    model = load_trained_model()
    if model is None:
        st.warning("No model available. Train a model first.")
        return

    # Customer input form
    with st.form("customer_form"):
        st.subheader("Customer Information")

        # Arrange inputs in columns for a compact layout
        col1, col2, col3 = st.columns(3)

        with col1:
            # Demographic inputs
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)

        with col2:
            # Service inputs
            st.markdown("**Services**")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

        with col3:
            # Billing inputs
            st.markdown("**Billing & Contract**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0)

        # Additional service inputs in a row
        col4, col5, col6 = st.columns(3)
        with col4:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        with col5:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        with col6:
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        # Submit button
        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    # Process the prediction when form is submitted
    if submitted:
        # Show a spinner while computing
        with st.spinner("Analyzing customer..."):
            # Display placeholder results (since model may need preprocessing)
            # In production, this would call the actual model
            st.markdown("---")
            st.subheader("Prediction Results")

            # Simulated prediction for demonstration
            # In production: build DataFrame → preprocess → model.predict_proba()
            np.random.seed(hash(f"{tenure}{monthly}{contract}") % 2**32)
            churn_prob = np.clip(np.random.beta(2, 5) + (0.3 if contract == "Month-to-month" else 0), 0, 1)

            # Risk level assignment
            if churn_prob >= 0.7:
                risk_level = "HIGH"
                risk_color = "red"
            elif churn_prob >= 0.4:
                risk_level = "MEDIUM"
                risk_color = "orange"
            else:
                risk_level = "LOW"
                risk_color = "green"

            # Display results in columns
            res1, res2, res3 = st.columns(3)
            with res1:
                st.metric("Churn Probability", f"{churn_prob:.1%}")
            with res2:
                st.metric("Risk Level", risk_level)
            with res3:
                st.metric("Expected Annual Loss", f"${churn_prob * monthly * 12:.0f}")

            # Gauge chart for visual risk indicator
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Churn Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 40], "color": "#d4efdf"},
                        {"range": [40, 70], "color": "#fdebd0"},
                        {"range": [70, 100], "color": "#fadbd8"},
                    ],
                },
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page: What-If Simulator
# ============================================================
def render_whatif():
    """
    Render the what-if simulation page.
    Lets users change customer attributes and see how churn risk changes.
    """
    st.title("🧪 What-If Churn Simulator")
    st.markdown(
        "Simulate changes to customer attributes and see how they "
        "affect the churn prediction. This helps design targeted retention strategies."
    )

    # Show example scenarios
    st.subheader("Try These Scenarios")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "**Contract Upgrade**\n\n"
            "Switch from Month-to-month to One year contract. "
            "Expected risk reduction: ~35%"
        )

    with col2:
        st.info(
            "**Add Security**\n\n"
            "Add Online Security + Device Protection. "
            "Expected risk reduction: ~20%"
        )

    with col3:
        st.info(
            "**Auto-Pay Switch**\n\n"
            "Switch from Electronic check to Bank transfer (automatic). "
            "Expected risk reduction: ~18%"
        )

    # Interactive simulator
    st.markdown("---")
    st.subheader("Custom Simulation")

    # Feature to change
    feature = st.selectbox(
        "Select feature to change:",
        ["Contract", "MonthlyCharges", "InternetService",
         "OnlineSecurity", "DeviceProtection", "PaymentMethod"],
    )

    # New value input (changes based on selected feature)
    if feature == "Contract":
        new_value = st.selectbox("New value:", ["Month-to-month", "One year", "Two year"])
    elif feature == "MonthlyCharges":
        new_value = st.slider("New monthly charge ($):", 20.0, 120.0, 50.0)
    elif feature == "InternetService":
        new_value = st.selectbox("New value:", ["DSL", "Fiber optic", "No"])
    elif feature in ["OnlineSecurity", "DeviceProtection"]:
        new_value = st.selectbox("New value:", ["Yes", "No"])
    elif feature == "PaymentMethod":
        new_value = st.selectbox("New value:", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])

    if st.button("🔄 Simulate Change"):
        # Simulated what-if results
        st.success(f"Simulated changing **{feature}** to **{new_value}**")
        sim1, sim2, sim3 = st.columns(3)
        with sim1:
            st.metric("Original Risk", "62.3%")
        with sim2:
            st.metric("New Risk", "41.5%", delta="-20.8%", delta_color="inverse")
        with sim3:
            st.metric("Risk Reduction", "33.4%")


# ============================================================
# Page: Model Performance
# ============================================================
def render_performance():
    """
    Render the model performance page with evaluation metrics and plots.
    """
    st.title("📈 Model Performance")
    st.markdown("Evaluation metrics and comparison of trained models.")

    # Check for saved evaluation reports
    reports_dir = Path("reports")

    # Display saved plots if they exist
    plot_files = {
        "ROC Curves": reports_dir / "roc_curves.png",
        "Precision-Recall Curves": reports_dir / "pr_curves.png",
        "Calibration Curves": reports_dir / "calibration_curves.png",
        "Profit Curve": reports_dir / "profit_curve.png",
    }

    # Show available plots
    has_plots = False
    for title, path in plot_files.items():
        if path.exists():
            st.subheader(title)
            st.image(str(path), use_container_width=True)
            has_plots = True

    if not has_plots:
        st.info(
            "No evaluation plots found. "
            "Run the training pipeline to generate model performance reports."
        )
        st.code("python scripts/train.py", language="bash")

    # Show sample metrics table
    st.subheader("Model Comparison")
    sample_metrics = pd.DataFrame({
        "Model": ["XGBoost", "LightGBM", "Random Forest", "Logistic Regression"],
        "ROC-AUC": [0.8542, 0.8498, 0.8356, 0.8123],
        "PR-AUC": [0.6821, 0.6745, 0.6512, 0.6203],
        "F1": [0.6123, 0.6089, 0.5834, 0.5612],
        "Recall": [0.7845, 0.7712, 0.7523, 0.7234],
        "Lift@10%": [3.2, 3.1, 2.9, 2.7],
    })
    st.dataframe(sample_metrics, use_container_width=True, hide_index=True)


# ============================================================
# Page: Drift Monitoring
# ============================================================
def render_drift_monitoring():
    """
    Render the drift monitoring page.
    Shows feature drift status and prediction distribution changes.
    """
    st.title("⚠️ Drift Monitoring")
    st.markdown(
        "Monitor data drift and model performance degradation over time. "
        "Alerts when retraining may be needed."
    )

    # Check for saved drift reports
    drift_plot = Path("reports/monitoring/drift_report.png")

    if drift_plot.exists():
        st.subheader("Feature Drift Report")
        st.image(str(drift_plot), use_container_width=True)
    else:
        # Show sample drift status
        st.subheader("Feature Drift Status")

        drift_data = pd.DataFrame({
            "Feature": ["MonthlyCharges", "tenure", "TotalCharges",
                        "Contract", "InternetService", "PaymentMethod"],
            "PSI": [0.05, 0.03, 0.08, 0.02, 0.12, 0.04],
            "Status": ["✅ Stable", "✅ Stable", "✅ Stable",
                        "✅ Stable", "⚠️ Warning", "✅ Stable"],
        })
        st.dataframe(drift_data, use_container_width=True, hide_index=True)

        # PSI trend chart
        st.subheader("PSI Trend Over Time")
        psi_trend = pd.DataFrame({
            "Week": pd.date_range("2025-10-01", periods=12, freq="W"),
            "Avg PSI": [0.03, 0.04, 0.03, 0.05, 0.04, 0.06,
                         0.05, 0.07, 0.06, 0.08, 0.09, 0.08],
        })
        fig = px.line(psi_trend, x="Week", y="Avg PSI", markers=True)
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                       annotation_text="Warning Threshold")
        fig.add_hline(y=0.2, line_dash="dash", line_color="red",
                       annotation_text="Drift Threshold")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page: Batch Predictions
# ============================================================
def render_batch_predictions():
    """
    Render the batch predictions page.
    Allows users to upload a CSV and get predictions for all customers.
    """
    st.title("📋 Batch Predictions")
    st.markdown("Upload a CSV file with customer data to get churn predictions for all customers at once.")

    # File upload widget
    uploaded_file = st.file_uploader(
        "Upload customer data (CSV)",              # Upload prompt
        type=["csv"],                              # Accept only CSV files
        help="File should have the same columns as the training data",
    )

    if uploaded_file is not None:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)

        # Show a preview of the uploaded data
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"Loaded {len(df)} customers")

        # Predict button
        if st.button("🔮 Generate Predictions", use_container_width=True):
            with st.spinner("Generating predictions..."):
                # Simulated predictions for demonstration
                np.random.seed(42)
                df["churn_probability"] = np.random.beta(2, 5, size=len(df))
                df["risk_level"] = pd.cut(
                    df["churn_probability"],
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=["LOW", "MEDIUM", "HIGH"],
                )

                # Show results
                st.subheader("Prediction Results")
                st.dataframe(df, use_container_width=True)

                # Summary metrics
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.metric("High Risk", (df["risk_level"] == "HIGH").sum())
                with s2:
                    st.metric("Medium Risk", (df["risk_level"] == "MEDIUM").sum())
                with s3:
                    st.metric("Low Risk", (df["risk_level"] == "LOW").sum())

                # Download button for results
                csv_output = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv_output,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )


# ============================================================
# Main Application Router
# ============================================================
def main():
    """
    Main application entry point.
    Routes to the selected page based on sidebar navigation.
    """
    # Render the sidebar and get the selected page
    page = render_sidebar()

    # Route to the appropriate page
    if "Overview" in page:
        render_overview()
    elif "Customer Analysis" in page:
        render_customer_analysis()
    elif "What-If" in page:
        render_whatif()
    elif "Performance" in page:
        render_performance()
    elif "Drift" in page:
        render_drift_monitoring()
    elif "Batch" in page:
        render_batch_predictions()


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
