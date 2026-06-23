"""Streamlit UI for the Telco churn scoring API.

The app calls the FastAPI service over HTTP (default: http://localhost:8000),
respecting the UI / scoring logic separation defined in the V2 specification.
"""

import requests
import streamlit as st


API_URL = "http://localhost:8000"


def collect_profile() -> dict:
    """Render the input form and return the customer profile as a dict.

    The returned dict matches the CustomerProfile schema expected by the API.
    """
    with st.form("customer_profile"):
        left, right = st.columns(2)

        with left:
            st.subheader("Customer profile")

            tenure = st.slider(
                "Tenure (months)", min_value=1, max_value=72, value=24,
                help="Number of months since subscription.",
            )
            MonthlyCharges = st.number_input(
                "Monthly charges ($)", min_value=0.0, max_value=200.0,
                value=70.0, step=5.0,
            )
            TotalCharges = st.number_input(
                "Total charges ($)", min_value=0.0, max_value=10000.0,
                value=1680.0, step=10.0,
            )

            SeniorCitizen_label = st.radio(
                "Senior citizen", options=["No", "Yes"], horizontal=True,
            )
            SeniorCitizen = 1 if SeniorCitizen_label == "Yes" else 0

            Partner = st.radio(
                "Has partner", options=["Yes", "No"], horizontal=True,
            )
            Dependents = st.radio(
                "Has dependents", options=["Yes", "No"], horizontal=True,
                index=1,
            )

        with right:
            st.subheader("Services and contract")

            InternetService = st.selectbox(
                "Internet service", options=["DSL", "Fiber optic", "No"],
            )

            # Internet-dependent option values depend on InternetService
            internet_options = (
                ["Yes", "No"] if InternetService != "No"
                else ["No internet service"]
            )

            nb_services = st.slider(
                "Number of active services", min_value=0, max_value=8, value=3,
                help="Total count of subscribed services (derived feature).",
            )

            MultipleLines = st.selectbox(
                "Multiple lines", options=["Yes", "No", "No phone service"],
                index=1,
            )
            OnlineSecurity = st.selectbox("Online security", options=internet_options)
            OnlineBackup = st.selectbox("Online backup", options=internet_options)
            DeviceProtection = st.selectbox("Device protection", options=internet_options)
            TechSupport = st.selectbox("Tech support", options=internet_options)
            StreamingTV = st.selectbox("Streaming TV", options=internet_options)
            StreamingMovies = st.selectbox("Streaming movies", options=internet_options)

            Contract = st.selectbox(
                "Contract type",
                options=["Month-to-month", "One year", "Two year"],
                index=1,
            )
            PaperlessBilling = st.radio(
                "Paperless billing", options=["Yes", "No"], horizontal=True,
            )
            PaymentMethod = st.selectbox(
                "Payment method",
                options=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=3,
            )

        submitted = st.form_submit_button("Score customer", type="primary")

    if not submitted:
        return {}

    return {
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "nb_services": nb_services,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "PaperlessBilling": PaperlessBilling,
        "MultipleLines": MultipleLines,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "InternetService": InternetService,
        "Contract": Contract,
        "PaymentMethod": PaymentMethod,
    }

def score_profile(profile: dict) -> dict | None:
    """Send the profile to the API and return the prediction dict.

    Returns None and displays a Streamlit error if the API is unreachable or
    returns a non-200 response.
    """
    try:
        response = requests.post(f"{API_URL}/predict", json=profile, timeout=5)
    except requests.ConnectionError:
        st.error(
            f"Cannot reach the API at {API_URL}. "
            "Make sure the Docker container is running (`docker ps`)."
        )
        return None
    except requests.Timeout:
        st.error("The API took too long to respond.")
        return None

    if response.status_code != 200:
        st.error(
            f"API returned status {response.status_code}: "
            f"{response.text}"
        )
        return None

    return response.json()

def render_prediction(prediction: dict) -> None:
    """Display the scoring result as KPIs and a short explanation."""
    proba = prediction["proba_churn"]
    label = prediction["label_pred"]
    segment = prediction["risk_segment"]
    threshold = prediction["threshold"]

    st.subheader("Scoring result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Churn probability", f"{proba:.1%}")
        st.progress(proba)
    with col2:
        verdict = "Churn risk" if label == 1 else "Low risk"
        st.metric("Decision", verdict, help=f"Threshold: {threshold:.3f}")
    with col3:
        st.metric("Risk segment", segment)

    st.markdown("**Key drivers**")
    drivers = []
    if proba >= 0.5:
        drivers.append("High predicted churn probability — this customer is a priority for retention actions.")
    elif proba >= threshold:
        drivers.append("Probability is above the decision threshold — eligible for retention offer.")
    else:
        drivers.append("Probability is below the decision threshold — no immediate action needed.")
    drivers.append(
        "The strongest drivers in this model are *Contract type* and *tenure* "
        "(short tenure + month-to-month contracts strongly increase churn risk)."
    )
    for line in drivers:
        st.write(f"- {line}")

def main() -> None:
    """Entrypoint of the Streamlit app."""
    st.set_page_config(
        page_title="Telco Churn Scoring",
        page_icon="",
        layout="wide",
    )

    st.title("Telco Churn Scoring")
    st.caption(
        "Churn scoring demo for TelcoWave. "
        "The UI calls the containerized FastAPI service."
    )

    st.divider()

    profile = collect_profile()

    if profile:
        st.divider()
        prediction = score_profile(profile)
        if prediction is not None:
            render_prediction(prediction)


if __name__ == "__main__":
    main()