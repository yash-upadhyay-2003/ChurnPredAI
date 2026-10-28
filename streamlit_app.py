"""
Streamlit frontend application for Customer Churn Prediction.
Provides a clean and simple interface for users to enter customer details
and receive real-time churn predictions from the FastAPI backend.
"""

import os
import requests
import streamlit as st

# Configure the Streamlit page layout
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📞",
    layout="centered"
)

# Read the backend API URL safely from Streamlit secrets (deployment) or environment variable (local/Render)
# Never hardcode a temporary or local-only URL
BACKEND_URL = None
try:
    if "BACKEND_URL" in st.secrets:
        BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    # On local environments without secrets.toml, st.secrets raises StreamlitSecretNotFoundError
    pass

if not BACKEND_URL:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

BACKEND_URL = BACKEND_URL.rstrip("/")

# Title and introductory description
st.title("📞 Customer Churn Prediction")
st.write(
    "Enter customer subscription, usage, and service details below to estimate "
    "the probability of customer churn using our trained Random Forest model."
)
st.divider()

# Sidebar: Service connection status and guidance
st.sidebar.header("Backend API Status")
st.sidebar.caption(f"Configured API URL: `{BACKEND_URL}`")

if st.sidebar.button("Check API Health"):
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if health_resp.status_code == 200:
            info = health_resp.json()
            st.sidebar.success("Backend is healthy and ready!")
            st.sidebar.write(f"**Model:** {info.get('selected_model')}")
            st.sidebar.write(f"**Threshold:** {info.get('decision_threshold'):.4f}")
        else:
            st.sidebar.error(f"Health check failed (Status {health_resp.status_code})")
    except Exception as err:
        st.sidebar.error(f"Cannot reach backend: {err}")

# Group 1: Subscription and Account Details
st.subheader("1. Subscription & Account Information")
col1, col2 = st.columns(2)

with col1:
    territory_options = [
        "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL",
        "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA",
        "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
        "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI",
        "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"
    ]
    territory = st.selectbox(
        "Territory (State)",
        options=territory_options,
        index=territory_options.index("CA"),
        help="State or territory where the customer account is registered."
    )

    plan_options = ["Purple Plan", "Red Plan", "Green Plan", "Blue Plan"]
    plan_type = st.selectbox(
        "Plan Type",
        options=plan_options,
        index=0,
        help="Selected service subscription plan."
    )

    promotion = st.selectbox(
        "Received Promotion?",
        options=["no", "yes"],
        index=0,
        help="Whether the customer enrolled through a special promotional offer."
    )

with col2:
    customer_tenure = st.number_input(
        "Customer Tenure (Months)",
        min_value=0,
        max_value=300,
        value=100,
        step=1,
        help="Total duration in months that the customer has been with the provider."
    )

    month_options = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    start_month = st.selectbox(
        "Contract Start Month",
        options=month_options,
        index=2,
        help="Calendar month when subscription started."
    )

    week_options = [f"W{i:02d}" for i in range(1, 54)]
    start_week = st.selectbox(
        "Contract Start Week",
        options=week_options,
        index=9,
        help="Week of the year when subscription started."
    )

st.divider()

# Group 2: Usage and Billing Metrics
st.subheader("2. Usage & Billing Metrics")
col3, col4 = st.columns(2)

with col3:
    base_fee = st.number_input(
        "Base Monthly Fee ($)",
        min_value=0.0,
        max_value=300.0,
        value=30.0,
        step=1.0,
        help="Standard monthly base charge for the account."
    )

    additional_spend = st.number_input(
        "Additional Feature Spend ($)",
        min_value=0.0,
        max_value=200.0,
        value=15.0,
        step=1.0,
        help="Monthly expenditure on add-on features and upgrades."
    )

with col4:
    current_usage = st.number_input(
        "Current Period Usage (Units)",
        min_value=0.0,
        max_value=2000.0,
        value=200.0,
        step=10.0,
        help="Usage volume registered in the current billing cycle."
    )

    prior_usage = st.number_input(
        "Prior Period Usage (Units)",
        min_value=0.0,
        max_value=2000.0,
        value=190.0,
        step=10.0,
        help="Usage volume registered in the previous billing cycle."
    )

st.divider()

# Group 3: Device and Customer Service Indicators
st.subheader("3. Device & Service Experience")
col5, col6 = st.columns(2)

with col5:
    device_options = ["Galaxy", "iPhone", "-"]
    device_type = st.selectbox(
        "Device Type",
        options=device_options,
        index=0,
        help="Device model family ('-' represents unrecorded device)."
    )

    service_rating = st.number_input(
        "Service Satisfaction Rating",
        min_value=0.0,
        max_value=20.0,
        value=10.0,
        step=0.5,
        help="Customer rating score given to customer service (0.0 to 20.0 scale)."
    )

with col6:
    service_tickets = st.number_input(
        "Service Tickets Raised",
        min_value=0,
        max_value=50,
        value=4,
        step=1,
        help="Total number of support tickets filed by the customer."
    )

    penalties = st.number_input(
        "Number of Penalties",
        min_value=0,
        max_value=30,
        value=1,
        step=1,
        help="Number of billing or service penalty infractions on record."
    )

st.divider()

# Prediction Submission Section
# The frontend sends entered raw inputs to FastAPI; it does not calculate features itself
if st.button("Predict Churn", type="primary", use_container_width=True):
    # Prepare the 14 raw feature payload
    payload = {
        "Territory": territory,
        "DeviceType": device_type,
        "Promotion": promotion,
        "PlanType": plan_type,
        "BaseFee": float(base_fee),
        "AdditionalFeatureSpend": float(additional_spend),
        "NumberOfPenalties": int(penalties),
        "CurrentPeriodUsage": float(current_usage),
        "PriorPeriodUsage": float(prior_usage),
        "ServiceRating": float(service_rating),
        "ServiceTickets": int(service_tickets),
        "StartMonth": start_month,
        "StartWeek": start_week,
        "CustomerTenure": int(customer_tenure)
    }

    # Show a progress spinner while the API call is processed
    with st.spinner("Communicating with FastAPI backend..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                churn_prob = result["churn_probability"]
                pred_label = result["predicted_label"]
                threshold = result["threshold_used"]

                # Display the prediction results prominently
                st.subheader("Prediction Result")
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(
                        label="Predicted Churn Probability",
                        value=f"{churn_prob * 100:.1f}%"
                    )
                with metric_col2:
                    st.metric(
                        label="Decision Threshold",
                        value=f"{threshold * 100:.1f}%"
                    )

                # Show clear outcome indicator
                if pred_label == "yes":
                    st.error(
                        f"**Outcome: High Churn Risk (Yes)**\n\n"
                        f"The estimated churn probability ({churn_prob * 100:.1f}%) "
                        f"meets or exceeds the optimal threshold ({threshold * 100:.1f}%)."
                    )
                else:
                    st.success(
                        f"**Outcome: Low Churn Risk (No)**\n\n"
                        f"The estimated churn probability ({churn_prob * 100:.1f}%) "
                        f"is below the decision threshold ({threshold * 100:.1f}%)."
                    )

            else:
                st.error(
                    f"Backend returned error status {response.status_code}: {response.text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                f"**Backend Connection Error:** Unable to reach FastAPI backend at `{BACKEND_URL}`.\n\n"
                "Please verify that the backend server is running and that your `BACKEND_URL` "
                "environment variable or Streamlit secret is correctly configured."
            )
        except requests.exceptions.Timeout:
            st.error("The request to the backend timed out after 10 seconds. Please try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
