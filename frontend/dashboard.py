import json

import requests
import streamlit as st


API_URL = "http://localhost:8000"


st.set_page_config(page_title="ClaimWatch AI Dashboard", layout="centered")
st.title("ClaimWatch AI – Fraud Detection Demo")
st.markdown(
    "Enter claim details below to get a fraud probability, anomaly score, "
    "and an auto-generated investigation summary."
)


with st.form("claim_form"):
    col1, col2 = st.columns(2)
    with col1:
        claim_amount = st.number_input("Claim Amount", min_value=0.0, value=5000.0, step=100.0)
        policy_tenure_days = st.number_input("Policy Tenure (days)", min_value=0, value=365, step=30)
    with col2:
        num_prior_claims = st.number_input("Number of Prior Claims", min_value=0, value=1, step=1)
        customer_age = st.number_input("Customer Age", min_value=18, value=35, step=1)

    submitted = st.form_submit_button("Evaluate Claim")


if submitted:
    payload = {
        "claim_amount": claim_amount,
        "policy_tenure_days": policy_tenure_days,
        "num_prior_claims": num_prior_claims,
        "customer_age": customer_age,
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            data = resp.json()
            st.subheader("Fraud Assessment")
            st.metric("Fraud Probability", f"{data['fraud_probability']:.2%}")
            st.metric("Anomaly Score", f"{data['anomaly_score']:.3f}")
            st.write(f"**Anomalous?** {'Yes' if data['is_anomalous'] else 'No'}")

            st.subheader("Explanation (Top Features)")
            st.table(
                [
                    {
                        "Feature": f["feature"],
                        "Value": f["value"],
                        "SHAP Impact": f["shap_value"],
                    }
                    for f in data["top_features"]
                ]
            )

            st.subheader("Investigation Summary")
            st.write(data["summary"])

            st.subheader("Recommended Next Actions")
            for action in data["recommended_actions"]:
                st.markdown(f"- {action}")

            with st.expander("Raw API Response"):
                st.code(json.dumps(data, indent=2), language="json")
    except Exception as exc:
        st.error(f"Request failed: {exc}")

