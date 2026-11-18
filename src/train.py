from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
import torch

from src.data import CLINICAL_FIELDS, DIAGNOSES, load_metadata, patient_folds
from src.evaluate import evaluate_predictions, save_plots
from src.images import load_image_features
from src.models import DEMOGRAPHICS, hide_fields, metadata_input, predict, train_model


def run_experiments(root):
    data = load_metadata(root / "data/metadata.csv")
    folds = patient_folds(data)
    features = load_image_features(data, root)
    labels = pd.Categorical(data["diagnostic"], categories=DIAGNOSES).codes
    diagnoses = np.array(DIAGNOSES)
    empty_images = np.empty((len(data), 0), dtype=np.float32)
    indicators = data[CLINICAL_FIELDS].isna().astype(int)
    metadata = metadata_input(data, CLINICAL_FIELDS)

    conditions = ["majority_class", "missingness_only", "image_only", "full_metadata", "demographics"]
    for cut in [0, 25, 50]:
        conditions.extend([f"fusion_standard_{cut}", f"fusion_masked_{cut}"])
    predictions = data[["img_id", "patient_id", "diagnostic"]].copy()
    predictions["fold"] = 0
    for name in conditions:
        predictions[name] = ""

    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    run_dir = root / "data/processed"
    losses = []
    for fold, (train, test) in enumerate(folds, start=1):
        seed = 42 + fold
        predictions.loc[test, "fold"] = fold
        print(f"Fold {fold}: {len(train)} training images, {len(test)} held out", flush=True)
        for name, model in [
            ("majority_class", DummyClassifier(strategy="most_frequent")),
            ("missingness_only", LogisticRegression(max_iter=1000)),
        ]:
            model.fit(indicators.iloc[train], labels[train])
            predictions.loc[test, name] = diagnoses[model.predict(indicators.iloc[test])]

        setups = [
            ("image_only", features, []),
            ("full_metadata", empty_images, CLINICAL_FIELDS),
            ("demographics", empty_images, DEMOGRAPHICS),
            ("fusion_standard", features, CLINICAL_FIELDS),
            ("fusion_masked", features, CLINICAL_FIELDS),
        ]
        demo_models = {}
        for name, images, fields in setups:
            values = metadata[fields]
            train_rate = 0.25 if name == "fusion_masked" else 0
            model, preprocessor, history = train_model(
                images[train], values.iloc[train], labels[train], len(DIAGNOSES), seed, train_rate
            )
            for epoch, loss in enumerate(history, start=1):
                losses.append({"fold": fold, "model": name, "epoch": epoch, "training_loss": loss})
            print(f"  {name}: loss {history[0]:.3f} -> {history[-1]:.3f}", flush=True)

            rates = [0, 25, 50] if name.startswith("fusion") else [0]
            for rate in rates:
                held_out = hide_fields(values.iloc[test], rate / 100, seed=1000 + fold)
                probabilities = predict(model, preprocessor, images[test], held_out)
                condition = f"{name}_{rate}" if name.startswith("fusion") else name
                predictions.loc[test, condition] = diagnoses[probabilities.argmax(axis=1)]

            if fold == 1 and name.startswith("fusion"):
                demo_models[name] = (model, preprocessor)
        if fold == 1:
            joblib.dump({
                "models": demo_models,
                "held_out_images": data.iloc[test]["img_id"].tolist(),
            }, run_dir / "fold1_demo.joblib")

    metrics = evaluate_predictions(data, predictions[conditions], results_dir)
    save_plots(data, predictions, metrics, results_dir)
    predictions.to_csv(run_dir / "prototype_predictions.csv", index=False)
    pd.DataFrame(losses).to_csv(results_dir / "training_loss.csv", index=False)
    return metrics


if __name__ == "__main__":
    torch.set_num_threads(4)
    project_root = Path(__file__).resolve().parents[1]
    results = run_experiments(project_root)
    print(results[results["metric"] == "balanced_accuracy"].round(4).to_string(index=False))
