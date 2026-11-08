"""CPU batches and resumable epoch-boundary checkpoints for the CNN reference."""

from pathlib import Path

import numpy as np
import torch

from music_genre.models.cnn import normalize_logmel


def epoch_order(positions, seed, epoch):
    generator = torch.Generator().manual_seed(seed + epoch)
    return np.asarray(positions)[torch.randperm(len(positions), generator=generator).numpy()]


def make_batch(matrix, positions, config):
    values = np.array(matrix[positions], dtype=np.float32, copy=True)
    if values.ndim != 3 or tuple(values.shape[1:]) != (128, 622) or not np.isfinite(values).all():
        raise ValueError("Expected finite native-resolution log-Mel windows")
    tensor = normalize_logmel(torch.from_numpy(values)[:, None], config["db_min"], config["db_max"])
    return tensor.contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def predict_windows(model, matrix, positions, config):
    model.eval()
    chunks = []
    for start in range(0, len(positions), config["batch_size"]):
        batch = make_batch(matrix, positions[start:start + config["batch_size"]], config)
        chunks.append(model(batch).softmax(dim=1).numpy())
    probabilities = np.concatenate(chunks)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite CNN probabilities")
    return probabilities


def save_checkpoint(path: Path, state):
    temporary = path.with_suffix(".partial.pt")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, identity):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["identity"] != identity:
        raise ValueError("Checkpoint configuration, data, software or source changed")
    return state
