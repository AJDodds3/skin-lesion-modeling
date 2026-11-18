import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


NUMERIC_FIELDS = ["age", "diameter_1", "diameter_2"]
DEMOGRAPHICS = ["age", "gender", "fitspatrick"]
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def metadata_input(data, fields):
    values = data[fields].copy()
    # Keep categorical fields in a consistent format, including boolean fields.
    for field in fields:
        if field not in NUMERIC_FIELDS:
            values[field] = values[field].map(str, na_action="ignore")
    return values


def metadata_preprocessor(values):
    numeric = [field for field in values if field in NUMERIC_FIELDS]
    categorical = [field for field in values if field not in NUMERIC_FIELDS]
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("missing", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_steps, numeric),
        ("categorical", categorical_steps, categorical),
    ])


def encode_metadata(preprocessor, values):
    encoded = preprocessor.transform(values)
    # Include an indicator even for fields that were complete during training.
    return np.column_stack([encoded, values.isna().to_numpy()]).astype(np.float32)


def hide_fields(values, fraction, seed):
    # A shared seed makes the 25% test mask a subset of the 50% mask.
    draws = np.random.default_rng(seed).random(values.shape)
    hidden = (draws < fraction) & values.notna().to_numpy(dtype=bool)
    return values.mask(hidden)


class Classifier(nn.Module):
    def __init__(self, image_size, metadata_size, classes):
        super().__init__()
        self.metadata = None
        if metadata_size:
            self.metadata = nn.Sequential(nn.Linear(metadata_size, 32), nn.ReLU())
        input_size = image_size + (32 if metadata_size else 0)
        self.output = nn.Linear(input_size, classes)

    def forward(self, images, metadata):
        features = images
        if self.metadata is not None:
            features = torch.cat([images, self.metadata(metadata)], dim=1)
        return self.output(features)


def train_model(images, values, labels, classes, seed, mask_rate=0):
    torch.manual_seed(seed)
    preprocessor = None
    metadata_size = 0
    if len(values.columns):
        preprocessor = metadata_preprocessor(values).fit(values)
        metadata_size = encode_metadata(preprocessor, values).shape[1]

    model = Classifier(images.shape[1], metadata_size, classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    counts = np.bincount(labels, minlength=classes)
    weights = len(labels) / (classes * counts)
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    images_tensor = torch.tensor(images, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    losses = []

    for epoch in range(EPOCHS):
        train_values = hide_fields(values, mask_rate, seed + epoch) if mask_rate else values
        metadata = np.empty((len(values), 0), dtype=np.float32)
        if preprocessor is not None:
            metadata = encode_metadata(preprocessor, train_values)

        dataset = TensorDataset(images_tensor, torch.from_numpy(metadata), labels_tensor)
        batches = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
        total_loss = 0.0
        for image_batch, metadata_batch, label_batch in batches:
            optimizer.zero_grad()
            predictions = model(image_batch, metadata_batch)
            loss = loss_function(predictions, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(label_batch)
        losses.append(total_loss / len(labels))

    return model.eval(), preprocessor, losses


def predict(model, preprocessor, images, values):
    metadata = np.empty((len(images), 0), dtype=np.float32)
    if preprocessor is not None:
        metadata = encode_metadata(preprocessor, values)
    with torch.inference_mode():
        scores = model(torch.tensor(images, dtype=torch.float32), torch.from_numpy(metadata))
        return scores.softmax(dim=1).numpy()
