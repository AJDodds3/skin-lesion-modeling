import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DIAGNOSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
CLINICAL_FIELDS = [
    "smoke", "drink", "background_father", "background_mother", "age",
    "pesticide", "gender", "skin_cancer_history", "cancer_history",
    "has_piped_water", "has_sewage_system", "fitspatrick", "region",
    "diameter_1", "diameter_2", "itch", "grew", "hurt", "changed",
    "bleed", "elevation",
]


def load_metadata(path):
    # Pandas already reads blank CSV cells as missing. UNK is also unavailable.
    data = pd.read_csv(path, na_values=["UNK"])

    keys = ["patient_id", "lesion_id", "img_id", "diagnostic"]
    required = keys + CLINICAL_FIELDS
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing metadata columns: {missing_columns}")
    if data[keys].isna().any().any():
        raise ValueError("Every row needs its patient, lesion, image, and diagnosis.")
    if data["img_id"].duplicated().any():
        raise ValueError("Each image ID should appear only once.")
    if set(data["diagnostic"]) != set(DIAGNOSES):
        raise ValueError("The metadata should contain all six PAD-UFES-20 diagnoses.")
    return data


def patient_folds(data):
    diagnoses = data["diagnostic"]
    patients = data["patient_id"]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    folds = []
    for train_rows, test_rows in splitter.split(data, diagnoses, groups=patients):
        train_patients = set(patients.iloc[train_rows])
        test_patients = set(patients.iloc[test_rows])
        if train_patients & test_patients:
            raise ValueError("A patient appears in both training and evaluation.")
        if diagnoses.iloc[test_rows].nunique() != len(DIAGNOSES):
            raise ValueError("An evaluation fold is missing one or more diagnoses.")
        folds.append((train_rows, test_rows))

    return folds
