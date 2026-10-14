
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Air Cargo Delay Prediction", page_icon="✈️")

st.title("✈️ Air Cargo Delivery Delay Prediction")
st.write("Enter planned shipment details to estimate delivery delay.")

pipeline = joblib.load("cargo_delay_pipeline.pkl")
preprocessor = pipeline.named_steps["preprocessor"]

features = list(preprocessor.feature_names_in_)
numeric_features = list(preprocessor.transformers_[0][2])
categorical_features = list(preprocessor.transformers_[1][2])

df = pd.read_csv("c2k_data_comma.csv", na_values="?")

with st.form("shipment_form"):
    values = {}

    st.subheader("Shipment Details")

    for feature in numeric_features:
        values[feature] = st.number_input(
            feature, value=0.0, format="%.2f"
        )

    for feature in categorical_features:
        options = sorted(df[feature].dropna().unique().tolist())
        options = [float(x) for x in options]

        if options:
            values[feature] = st.selectbox(feature, options)
        else:
            values[feature] = 0.0

    submitted = st.form_submit_button("Predict Delay")

if submitted:
    shipment = pd.DataFrame([values], columns=features)
    prediction = pipeline.predict(shipment)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("The model predicts that this shipment may be DELAYED.")
    else:
        st.success("The model predicts that this shipment may be ON-TIME.")

    st.caption("This is a machine-learning estimate, not a guarantee.")
