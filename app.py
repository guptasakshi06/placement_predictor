from csv import DictReader
from pathlib import Path

import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).with_name("placement.csv")


@st.cache_resource
def train_model():
    """Train one cached preprocessing-and-classification pipeline."""
    features = []
    targets = []

    with DATA_PATH.open(encoding="utf-8", newline="") as data_file:
        rows = DictReader(data_file)
        for row in rows:
            # The training notebook uses this exact feature order: CGPA, then IQ.
            features.append([float(row["cgpa"]), float(row["iq"])])
            targets.append(int(row["placement"]))

    if not features:
        raise ValueError("placement.csv does not contain any training rows")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    return model.fit(features, targets)


model = train_model()

st.title("Placement Predictor")
st.caption("Estimate placement likelihood from a student's CGPA and IQ.")

cgpa = st.number_input(
    "Enter CGPA",
    min_value=0.0,
    max_value=10.0,
    value=6.0,
    step=0.1,
)
iq = st.number_input(
    "Enter IQ",
    min_value=0.0,
    value=100.0,
    step=1.0,
)

if st.button("Predict", type="primary"):
    # Prediction must use the same feature order used during training.
    prediction = model.predict([[cgpa, iq]])[0]

    if prediction == 1:
        st.success("The student is likely to get placed")
    else:
        st.error("The student is unlikely to get placed")
