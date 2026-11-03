
# Real Estate Investment Advisor
# Classification is based on a relative price benchmark.
# It does not establish future returns or profitability.

import json
import joblib
import pandas as pd
import streamlit as st

# Configure the application page
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

# Load model and metadata
@st.cache_resource
def load_resources():
    # Load the trained classification pipeline
    model = joblib.load(
        "random_forest_investment_model.pkl"
    )

    # Load feature definitions and input choices
    with open("metadata.json", "r") as file:
        metadata = json.load(file)

    return model, metadata

model, metadata = load_resources()

# Application heading
st.title("🏡 Real Estate Investment Advisor")
st.write(
    "Explore a property's relative price benchmark "
    "and an assumption-based price scenario."
)

# Explain the target and model limitations
st.warning(
    "The investment classification identifies properties "
    "that are relatively cheaper than a City + Property_Type "
    "training benchmark. It does NOT predict actual investment "
    "returns, profitability, or future price appreciation."
)

# Collect property details
st.header("Property Details")

# Create a form to gather user inputs
with st.form("property_form"):
    user_inputs = {}

    # Display categorical inputs
    st.subheader("Property Characteristics")

    for col in metadata["categorical_features"]:
        options = metadata["category_options"].get(col, [])

        # Skip fields that have no available options
        if not options:
            continue

        # Use the first known category as a default
        user_inputs[col] = st.selectbox(
            col.replace("_", " "),
            options=options
        )

    # Display numerical inputs
    st.subheader("Numerical Characteristics")

    for col in metadata["numerical_features"]:
        limits = metadata["numerical_ranges"][col]

        # Use dataset bounds and median as sensible defaults
        min_value = float(limits["min"])
        max_value = float(limits["max"])
        default_value = float(limits["median"])

        # Handle a constant numerical feature
        if min_value == max_value:
            user_inputs[col] = min_value
        else:
            user_inputs[col] = st.number_input(
                col.replace("_", " "),
                min_value=min_value,
                max_value=max_value,
                value=min(
                    max(default_value, min_value),
                    max_value
                )
            )

    # Scenario inputs are assumptions, not model predictions
    st.subheader("Optional Price Scenario")

    purchase_price = st.number_input(
        "Assumed Purchase Price (₹ Lakhs)",
        min_value=0.1,
        value=100.0,
        step=1.0
    )

    annual_growth = st.number_input(
        "Assumed Annual Price Growth (%)",
        min_value=-100.0,
        max_value=100.0,
        value=5.0,
        step=0.5
    )

    # Submit property information
    submitted = st.form_submit_button(
        "Analyze Property"
    )

# Process submitted information
if submitted:

    # Arrange input features in the exact training order
    input_df = pd.DataFrame(
        [[user_inputs[col] for col in metadata["feature_names"]]],
        columns=metadata["feature_names"]
    )

    # Predict the relative benchmark class
    prediction = int(model.predict(input_df)[0])

    # Display the classification result
    st.header("Analysis Result")

    if prediction == 1:
        st.success(
            "The property is classified as relatively cheaper "
            "than its training benchmark."
        )
    else:
        st.info(
            "The property is classified as at or above "
            "its training benchmark."
        )

    # Explain the meaning of the prediction
    st.caption(
        "This is a relative price classification, not a "
        "recommendation to buy or a guarantee of returns."
    )

    # Calculate a five-year scenario using the user's assumption
    future_scenario = (
        purchase_price * (1 + annual_growth / 100) ** 5
    )

    st.subheader("Five-Year Illustrative Price Scenario")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Assumed Purchase Price",
            f"₹{purchase_price:,.2f} Lakhs"
        )

    with col2:
        st.metric(
            "Scenario Price After 5 Years",
            f"₹{future_scenario:,.2f} Lakhs"
        )

    # Clarify that this is not a regression forecast
    st.info(
        f"This scenario assumes {annual_growth:.1f}% annual "
        "growth for five years. It is a mathematical illustration, "
        "not a model-generated forecast or guaranteed outcome. "
        "It excludes transaction costs, taxes, maintenance, "
        "rental income, and market uncertainty."
    )

# Footer
st.divider()
st.caption(
    "Educational prototype | Validate property details and "
    "conduct independent due diligence before investing."
)
