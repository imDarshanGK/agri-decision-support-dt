"""
app.py
------
Streamlit demo app for the crop recommendation system.
Run with: streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

from src.rank_crops import rank_crops, FEATURE_COLS
from src.explain import extract_decision_rules

st.set_page_config(page_title="Crop Recommendation - Decision Support", layout="centered")

st.title("Crop Recommendation Decision Support System")
st.write(
    "Optimized Decision Tree model — enter your soil and climate parameters "
    "to get ranked crop recommendations with confidence scores."
)


@st.cache_resource
def get_model():
    return joblib.load("models/optimized_dt.pkl")


model = get_model()

st.header("Input Parameters")
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=42.0)
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=43.0)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=20.8)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=202.9)

top_k = st.slider("Number of crop recommendations", min_value=1, max_value=5, value=3)

if st.button("Get Recommendations"):
    input_dict = {
        "N": N, "P": P, "K": K, "temperature": temperature,
        "humidity": humidity, "ph": ph, "rainfall": rainfall,
    }

    results = rank_crops(model, input_dict, top_k=top_k)

    st.header("Recommended Crops")
    for i, r in enumerate(results, start=1):
        st.write(f"**{i}. {r['crop']}** — confidence: {r['confidence']*100:.1f}%")
        st.progress(r["confidence"])

    with st.expander("Why these crops? (Feature importance)"):
        importances = pd.Series(
            model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)
        st.bar_chart(importances)

    with st.expander("View underlying decision rules"):
        rules_text = extract_decision_rules(model)
        st.text(rules_text[:3000] + ("..." if len(rules_text) > 3000 else ""))

st.markdown("---")
st.caption("Decision Tree Optimization Framework for Advanced Decision Support and Business Analytics")
