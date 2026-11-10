from pathlib import Path

import numpy as np
import pytest
import torch

from music_genre.models.vit_native import NativeViT, ast_positions

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "artifacts/pretrained/vit_tiny"


def coordinate_grid():
    rows, columns = torch.meshgrid(torch.arange(14.0), torch.arange(14.0), indexing="ij")
    return torch.stack([rows, columns])[None]  # (1, 2, 14, 14): channel 0 = row, 1 = column


def test_positions_cut_frequency_from_the_middle_and_stretch_time_like_ast():
    out = ast_positions(coordinate_grid(), 8, 38)
    assert out.shape == (1, 8 * 38, 2)
    rows, columns = out[0, :, 0].view(8, 38), out[0, :, 1].view(8, 38)
    assert torch.equal(rows[:, 0], torch.arange(3.0, 11.0))  # rows 3..10 kept, row-major tokens
    assert torch.all(rows == rows[:, :1])
    assert columns[0, 0] == 0 and columns[0, -1] == 13 and torch.all(columns.diff(dim=1) >= 0)
    assert torch.all(columns == columns[:1])


def test_positions_are_unchanged_on_the_pretrained_grid():
    assert torch.equal(ast_positions(coordinate_grid(), 14, 14), coordinate_grid().flatten(2).transpose(1, 2))


@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").exists(), reason="local ViT weights absent")
def test_native_vit_matches_v1_blocks_on_a_224_grid_and_accepts_the_native_grid():
    from music_genre.models.vit import FrozenViT
    rng = np.random.default_rng(0)
    square = rng.uniform(-80, 20, (224, 224)).astype(np.float32)
    native = NativeViT(WEIGHTS)
    both = native.extract(square, -80.0, 20.0)
    assert both.shape == (384,)
    for half, block in ((both[:192], 6), (both[192:], 12)):
        np.testing.assert_allclose(half, FrozenViT(WEIGHTS, block).extract(square, -80.0, 20.0), atol=1e-5)
    wide = native.extract(rng.uniform(-80, 20, (128, 622)).astype(np.float32), -80.0, 20.0)
    assert wide.shape == (384,) and np.isfinite(wide).all()
