
"""Streamlit interface for the trained concrete compressive strength model."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from src.data_utils import FEATURES, TARGET, load_dataset

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_concrete_strength_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️", layout="wide")
st.title("🏗️ Concrete Compressive Strength Predictor")
st.caption("Educational regression model using the UCI Concrete Compressive Strength dataset.")
st.warning("For learning and preliminary estimation only. Do not use this tool for structural design, code compliance, or safety-critical decisions without qualified engineering review and laboratory testing.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python src/train.py` first, then restart the app.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
st.write(f"Selected model: **{metadata.get('best_model', 'Saved model')}**")

defaults = {
    "Cement": 300.0, "Blast Furnace Slag": 100.0, "Fly Ash": 0.0,
    "Water": 180.0, "Superplasticizer": 5.0, "Coarse Aggregate": 1000.0,
    "Fine Aggregate": 750.0, "Age": 28.0
}
ranges = {
    "Cement": (0.0, 600.0), "Blast Furnace Slag": (0.0, 400.0),
    "Fly Ash": (0.0, 300.0), "Water": (100.0, 250.0),
    "Superplasticizer": (0.0, 35.0), "Coarse Aggregate": (700.0, 1200.0),
    "Fine Aggregate": (500.0, 1000.0), "Age": (1.0, 365.0)
}
with st.form("mix_form"):
    st.subheader("Enter concrete mix proportions")
    st.caption("Material inputs are in kg/m³. Age is in days. The slider bounds are convenient input controls, not design-code limits.")
    cols = st.columns(2)
    values = {}
    for i, feature in enumerate(FEATURES):
        lo, hi = ranges[feature]
        step = 1.0 if feature == "Age" else 1.0
        with cols[i % 2]:
            values[feature] = st.number_input(
                feature + (" (days)" if feature == "Age" else " (kg/m³)"),
                min_value=float(lo), max_value=float(hi),
                value=float(defaults[feature]), step=step
            )
    submitted = st.form_submit_button("Predict compressive strength", type="primary")

if submitted:
    input_df = pd.DataFrame([values], columns=FEATURES)
    prediction = float(model.predict(input_df)[0])
    st.divider()
    st.metric("Predicted compressive strength", f"{prediction:.2f} MPa")
    st.info("This is a point estimate. The app does not calculate a validated confidence interval or replace cube/cylinder tests.")
    with st.expander("Review the submitted mix"):
        st.dataframe(input_df, use_container_width=True)
    st.download_button("Download input as CSV", input_df.to_csv(index=False), "concrete_mix_input.csv", "text/csv")

with st.expander("About this project"):
    st.markdown("""
    - **Task:** Regression
    - **Dataset:** UCI Concrete Compressive Strength, 1,030 records
    - **Inputs:** Seven material proportions and curing age
    - **Output:** Compressive strength in MPa
    - **Evaluation:** MAE, MSE, RMSE and R² on a held-out test set
    - **Limitations:** Dataset domain, measurement conditions, mix-design coverage and random-split evaluation constrain generalisation.
    """)
