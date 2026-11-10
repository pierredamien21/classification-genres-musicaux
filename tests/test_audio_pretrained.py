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


def test_maest_input_matches_the_original_maest_melspectrogram_inside_the_window():
    """palonso/MAEST models/helpers/melspectrogram.py (torchaudio Spectrogram + slaney MelScale), rewritten
    with torch.stft and librosa. Edge frames differ: MAEST pads by reflection, the HF port with zeros."""
    import librosa
    config = {"n_fft": 512, "hop_length": 256, "num_mel_bins": 96, "sampling_rate": 16000,
              "max_length": 626, "mean": 2.06755686098554, "std": 1.268292820667291}
    x = np.random.default_rng(0).normal(0, 0.1, 160000)
    power = torch.stft(torch.tensor(x), 512, 256, window=torch.hann_window(512, dtype=torch.float64),
                       center=True, pad_mode="reflect", return_complex=True).abs() ** 2
    bank = torch.tensor(librosa.filters.mel(sr=16000, n_fft=512, n_mels=96, norm="slaney"), dtype=torch.float64)
    original = ((torch.log10(1 + (bank @ power) * 10000) - config["mean"]) / (config["std"] * 2)).T.numpy()
    ours = maest_input(x.astype(np.float32), config)[0]
    np.testing.assert_allclose(ours[1:-1], original[1:-1], atol=1e-5)


@pytest.mark.filterwarnings("ignore:At least one mel filter")  # one Kaldi band is empty at 16 kHz, as in Kaldi
def test_ast_extractor_without_torchaudio_matches_a_kaldi_fbank():
    """Kaldi fbank defaults used by AST (25 ms / 10 ms, DC removal, preemphasis 0.97, symmetric Hann,
    512-point FFT, 128 Kaldi mel bands from 20 Hz over bins 0..255, log with float32 epsilon)."""
    from transformers import ASTFeatureExtractor
    x = np.random.default_rng(0).normal(0, 0.1, 160000)
    frames = np.stack([x[i * 160:i * 160 + 400] for i in range(998)])
    frames = frames - frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1].copy()
    frames[:, 0] *= 0.03
    frames *= 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)
    power = np.abs(np.fft.rfft(frames, 512)) ** 2
    mel = 1127 * np.log(1 + np.arange(256) * 16000 / 512 / 700)
    low, high = 1127 * np.log(1 + 20 / 700), 1127 * np.log(1 + 8000 / 700)
    step = (high - low) / 129
    bank = np.zeros((128, 257))
    for b in range(128):
        left, center, right = low + b * step, low + (b + 1) * step, low + (b + 2) * step
        rise, fall = (mel - left) / (center - left), (right - mel) / (right - center)
        bank[b, :256] = np.where((mel > left) & (mel < right), np.where(mel <= center, rise, fall), 0)
    kaldi = (np.log(np.maximum(power @ bank.T, np.finfo(np.float32).eps)) + 4.2677393) / (4.5689974 * 2)
    values = ASTFeatureExtractor()(x.astype(np.float32), sampling_rate=16000, return_tensors="np")["input_values"][0]
    assert values.shape == (1024, 128)
    np.testing.assert_allclose(values[:998], kaldi, atol=1e-5)
    assert np.allclose(values[998:], 4.2677393 / (4.5689974 * 2))  # AST pads the fbank with zeros, then normalizes
