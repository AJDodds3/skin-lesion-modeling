from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data import CLINICAL_FIELDS, DIAGNOSES, load_metadata
from src.images import image_paths
from src.models import hide_fields, metadata_input, predict


DIAGNOSIS_NAMES = [
    "ACK · Actinic keratosis",
    "BCC · Basal cell carcinoma",
    "MEL · Melanoma",
    "NEV · Nevus",
    "SCC · Squamous cell carcinoma",
    "SEK · Seborrheic keratosis",
]
MODEL_NAMES = {
    "fusion_standard": "Standard training",
    "fusion_masked": "25% masking during training",
}


@st.cache_resource
def load_run():
    processed = root / "data/processed"
    data = load_metadata(root / "data/metadata.csv")
    demo = joblib.load(processed / "fold1_demo.joblib")
    with np.load(processed / "resnet18_features.npz") as cache:
        if not np.array_equal(cache["image_ids"], data["img_id"].to_numpy(dtype=str)):
            raise ValueError("The cached image features do not match the current metadata.")
        features = cache["features"]
    examples = data[data["img_id"].isin(demo["held_out_images"])]
    paths = image_paths(root / "data/images", examples["img_id"])
    return examples, features, paths, demo["models"]


st.set_page_config(page_title="Skin lesion metadata experiment", layout="wide")
st.title("Skin lesion metadata experiment")
st.caption("Research demo using held-out PAD-UFES-20 cases from fold 1. Model scores are for coursework and research.")

try:
    examples, features, paths, models = load_run()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.info("Place the image archives and metadata in data/, then run python -m src.train.")
    st.stop()

controls, results = st.columns([1, 1.55], gap="large")
with controls:
    st.subheader("Held-out case")
    diagnosis = st.selectbox("Recorded diagnosis", DIAGNOSES, format_func=lambda code: DIAGNOSIS_NAMES[DIAGNOSES.index(code)])
    case_rows = examples.index[examples["diagnostic"] == diagnosis].tolist()
    case_number = st.selectbox("Example", range(1, len(case_rows) + 1))
    row = case_rows[case_number - 1]
    case = examples.loc[[row]]
    st.image(str(paths[case["img_id"].iloc[0]]), width=285)

    values = metadata_input(case, CLINICAL_FIELDS)
    recorded = values.columns[values.notna().iloc[0]].tolist()
    rate = st.select_slider("Additional fields to hide", options=[0, 25, 50], format_func=lambda n: f"{n}%")
    fields = st.multiselect("Hide specific recorded fields", recorded, format_func=lambda field: field.replace("_", " "))
    visible = hide_fields(values, rate / 100, seed=1000 + row)
    visible.loc[:, fields] = np.nan
    hidden = values.notna() & visible.isna()
    st.caption(f"{len(recorded)} recorded fields · {int(hidden.iloc[0].sum())} hidden for this comparison")
    with st.expander("Field availability"):
        availability = pd.Series("Recorded", index=CLINICAL_FIELDS)
        availability.loc[values.isna().iloc[0]] = "Missing in dataset"
        availability.loc[hidden.iloc[0]] = "Hidden for this run"
        st.dataframe(availability.rename("Status"), width="stretch")

with results:
    st.subheader("Predicted diagnosis scores")
    scores = pd.DataFrame(index=DIAGNOSIS_NAMES)
    for name, (model, preprocessor) in models.items():
        probabilities = predict(model, preprocessor, features[[row]], visible)
        scores[MODEL_NAMES[name]] = probabilities[0]
    for column, container in zip(scores.columns, st.columns(2)):
        container.metric(column, scores[column].idxmax().split(" · ")[0],
                         f"{scores[column].max():.1%}", delta_color="off", delta_arrow="off")
    chart = scores.copy()
    chart.index = DIAGNOSES
    chart.columns = ["Standard", "Masked training"]
    st.bar_chart(chart, horizontal=True, x_label="Diagnosis", y_label="Model score", stack=False)
    st.dataframe(scores.style.format("{:.1%}"), width="stretch")

with st.expander("Evaluation across all five folds"):
    metrics_path = root / "results/prototype_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        table = metrics[metrics["metric"] == "balanced_accuracy"].drop(columns="metric")
        st.dataframe(table, hide_index=True, width="stretch")
