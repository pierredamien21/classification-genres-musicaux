import numpy as np
import pytest
import torch

from music_genre.models.audio_pretrained import maest_input, transformer_vector


def test_transformer_vector_takes_cls_dist_and_patch_mean_at_block_6_and_final_norm():
    hidden = [torch.full((1, 4, 2), float(layer)) for layer in range(13)]
    hidden[6][0, 2:] = torch.tensor([[1.0, 3.0], [3.0, 5.0]])
    final = torch.full((1, 4, 2), -1.0)
    vector = transformer_vector(hidden, final)
    assert vector.tolist() == [6, 6, 6, 6, 2, 4, -1, -1, -1, -1, -1, -1]


def test_maest_input_is_626_frames_of_96_bands_for_ten_seconds():
    config = {"n_fft": 512, "hop_length": 256, "num_mel_bins": 96, "sampling_rate": 16000,
              "max_length": 626, "mean": 2.06755686098554, "std": 1.268292820667291}
    values = maest_input(np.zeros(160000), config)
    assert values.shape == (1, 626, 96)
    assert np.allclose(values, -config["mean"] / (2 * config["std"]), atol=1e-6)


def test_pretrained_features_reject_unknown_models(tmp_path):
    from music_genre.features.pretrained import PretrainedAudio
    with pytest.raises(ValueError):
        PretrainedAudio({"model": "musicfm"}, tmp_path)
