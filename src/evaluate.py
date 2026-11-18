import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.data import DIAGNOSES


def scores(matrices):
    correct = np.diagonal(matrices, axis1=-2, axis2=-1)
    actual = matrices.sum(axis=-1)
    predicted = matrices.sum(axis=-2)
    recall = np.divide(correct, actual, out=np.zeros_like(correct, dtype=float), where=actual > 0)
    totals = actual + predicted
    f1 = np.divide(2 * correct, totals, out=np.zeros_like(recall), where=totals > 0)
    overall = np.stack([
        correct.sum(axis=-1) / actual.sum(axis=-1),
        recall.mean(axis=-1),
        f1.mean(axis=-1),
    ], axis=-1)
    return np.concatenate([overall, recall], axis=-1)


def patient_confusion(data, predicted):
    patients = pd.factorize(data["patient_id"])[0]
    actual = pd.Categorical(data["diagnostic"], categories=DIAGNOSES).codes
    predicted = pd.Categorical(predicted, categories=DIAGNOSES).codes
    if (predicted < 0).any():
        raise ValueError("Every evaluation image needs a valid predicted diagnosis.")

    counts = np.zeros((len(np.unique(patients)), len(DIAGNOSES), len(DIAGNOSES)), dtype=int)
    np.add.at(counts, (patients, actual, predicted), 1)
    return counts


def evaluate_predictions(data, predictions, results_dir):
    names = ["accuracy", "balanced_accuracy", "macro_f1"]
    for diagnosis in DIAGNOSES:
        names.append(f"recall_{diagnosis}")

    number_of_patients = data["patient_id"].nunique()
    rng = np.random.default_rng(42)
    # Each draw samples whole patients and includes all of their images.
    bootstrap_weights = rng.multinomial(
        number_of_patients, np.full(number_of_patients, 1 / number_of_patients), size=2000
    )
    point_estimates = {}
    bootstrap_estimates = {}
    rows = []
    for name, predicted in predictions.items():
        counts = patient_confusion(data, predicted)
        matrix = counts.sum(axis=0)
        point = scores(matrix)
        boot_matrices = (bootstrap_weights @ counts.reshape(number_of_patients, -1)).reshape(-1, 6, 6)
        boot = scores(boot_matrices)
        point_estimates[name] = point
        bootstrap_estimates[name] = boot

        lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
        for metric, estimate, low, high in zip(names, point, lower, upper):
            rows.append({
                "condition": name, "metric": metric, "estimate": estimate,
                "ci_low": low, "ci_high": high,
            })

    intervals = pd.DataFrame(rows)
    intervals.to_csv(results_dir / "prototype_metrics.csv", index=False)

    missingness = point_estimates["missingness_only"] - point_estimates["majority_class"]
    missingness_boot = bootstrap_estimates["missingness_only"] - bootstrap_estimates["majority_class"]
    contrasts = [("missingness_vs_majority", missingness, missingness_boot)]
    for rate in [0, 25, 50]:
        standard = f"fusion_standard_{rate}"
        masked = f"fusion_masked_{rate}"
        difference = point_estimates[masked] - point_estimates[standard]
        boot_difference = bootstrap_estimates[masked] - bootstrap_estimates[standard]
        contrasts.append((f"masked_minus_standard_{rate}", difference, boot_difference))
        if rate:
            # Compare each model's loss of performance from its own 0% result.
            clean_difference = point_estimates["fusion_masked_0"] - point_estimates["fusion_standard_0"]
            clean_boot = bootstrap_estimates["fusion_masked_0"] - bootstrap_estimates["fusion_standard_0"]
            contrasts.append((f"smaller_drop_{rate}", difference - clean_difference, boot_difference - clean_boot))

    paired_rows = []
    for name, point, boot in contrasts:
        lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
        for index in [1, 2]:
            paired_rows.append({
                "comparison": name, "metric": names[index], "difference": point[index],
                "ci_low": lower[index], "ci_high": upper[index],
            })
    pd.DataFrame(paired_rows).to_csv(results_dir / "prototype_paired_differences.csv", index=False)
    return intervals


def save_plots(data, predictions, metrics, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4), layout="constrained")
    for model, label, ax in zip(
        ["missingness_only", "fusion_standard_0"],
        ["Missingness only", "Image + metadata (0% hidden)"],
        axes,
    ):
        matrix = patient_confusion(data, predictions[model]).sum(axis=0)
        ConfusionMatrixDisplay(matrix, display_labels=DIAGNOSES).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(label)
    fig.savefig(results_dir / "prototype_confusion_matrices.png", dpi=180)
    plt.close(fig)

    balanced = metrics[metrics["metric"] == "balanced_accuracy"].set_index("condition")
    fig, ax = plt.subplots(figsize=(6.2, 3.9), layout="constrained")
    for model in ["standard", "masked"]:
        rows = balanced.loc[[f"fusion_{model}_{rate}" for rate in [0, 25, 50]]]
        label = "Standard training" if model == "standard" else "25% masking during training"
        ax.plot([0, 25, 50], rows["estimate"], marker="o", label=label)
        ax.fill_between([0, 25, 50], rows["ci_low"], rows["ci_high"], alpha=0.13)
    ax.set(xticks=[0, 25, 50], xlabel="Additional clinical fields hidden at evaluation (%)",
           ylabel="Balanced accuracy")
    ax.legend()
    fig.savefig(results_dir / "masking_comparison.png", dpi=180)
    plt.close(fig)
