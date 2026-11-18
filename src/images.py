from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.models import ResNet18_Weights, resnet18


def image_paths(image_dir, image_ids):
    expected = set(image_ids)
    found = {}
    for path in sorted(Path(image_dir).rglob("*.png")):
        if path.name in expected:
            if path.name in found:
                raise ValueError(f"Duplicate image in data/images: {path.name}")
            found[path.name] = path

    missing = expected - found.keys()
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} PNGs in {image_dir}. Extract all three image archives there."
        )
    return found


def image_features(paths, image_ids):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet18_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model = model.to(device).eval()

    features = []
    batch_size = 32
    print(f"Extracting ResNet-18 features on {device}")
    with torch.inference_mode():
        for start in range(0, len(image_ids), batch_size):
            batch = []
            for image_id in image_ids[start:start + batch_size]:
                with Image.open(paths[image_id]) as image:
                    batch.append(transform(image.convert("RGB")))
            embeddings = model(torch.stack(batch).to(device))
            features.append(embeddings.cpu().numpy())
            if len(features) % 10 == 0 or start + batch_size >= len(image_ids):
                print(f"  {min(start + batch_size, len(image_ids))}/{len(image_ids)} images", flush=True)

    return np.concatenate(features)


def load_image_features(data, root):
    ids = data["img_id"].to_numpy(dtype=str)
    paths = image_paths(root / "data/images", ids)
    cache = root / "data/processed/resnet18_features.npz"
    if cache.exists():
        with np.load(cache) as saved:
            if np.array_equal(saved["image_ids"], ids) and saved["features"].shape == (len(ids), 512):
                print("Loaded cached ResNet-18 features.")
                return saved["features"]

    features = image_features(paths, ids)
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, image_ids=ids, features=features)
    return features
