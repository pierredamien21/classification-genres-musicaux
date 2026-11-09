"""Frozen audio-pretrained encoders for V2 linear probes: AST, MAEST and PANNs CNN14 (16 kHz).

No downloaded code is executed. AST uses the transformers built-in feature extractor. MAEST's
input follows feature_extraction_maest.py (Apache-2.0, read and re-implemented here with the
same transformers.audio_utils calls). CNN14 follows the PANNs architecture (Kong et al., 2020);
its STFT and mel filters are the frozen tensors stored in the checkpoint.
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TRANSFORMER_LAYERS = (6, 12)


def transformer_vector(hidden_states, last_hidden_state):
    """[CLS, DIST, mean of patch tokens] at the middle block and after the final norm."""
    parts = []
    for layer in TRANSFORMER_LAYERS:
        tokens = last_hidden_state if layer == len(hidden_states) - 1 else hidden_states[layer]
        tokens = tokens[0]
        parts += [tokens[0], tokens[1], tokens[2:].mean(dim=0)]
    return torch.cat(parts).numpy().astype(np.float64)


def maest_input(samples, config):
    """MAEST log-compressed mel input: 96 Slaney bands, hop 256, padded to max_length frames."""
    from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

    mel = mel_filter_bank(num_frequency_bins=config["n_fft"] // 2 + 1, num_mel_filters=config["num_mel_bins"],
                          min_frequency=0, max_frequency=config["sampling_rate"] / 2,
                          sampling_rate=config["sampling_rate"], norm="slaney", mel_scale="slaney")
    spec = spectrogram(np.asarray(samples, dtype=np.float32), window=window_function(config["n_fft"], "hann"),
                       frame_length=config["n_fft"], hop_length=config["hop_length"], power=2, mel_filters=mel,
                       min_value=1e-30, mel_floor=1e-30, pad_mode="constant").T
    spec = torch.Tensor(np.log10(1 + spec * 10000))
    difference = config["max_length"] - spec.shape[0]
    spec = F.pad(spec, (0, 0, 0, difference)) if difference > 0 else spec[:config["max_length"]]
    return ((spec.numpy() - config["mean"]) / (config["std"] * 2))[None]


class TransformerEncoder:
    """AST or MAEST (ASTForAudioClassification weights), frozen, CPU, evaluation mode."""

    def __init__(self, directory: Path, kind: str):
        import json

        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.kind = kind
        self.model = ASTForAudioClassification.from_pretrained(directory, local_files_only=True).eval()
        self.preprocessor = json.loads((directory / "preprocessor_config.json").read_text())
        if kind == "ast":
            self.extractor = ASTFeatureExtractor.from_pretrained(directory, local_files_only=True)
        elif kind != "maest":
            raise ValueError(f"Unknown transformer kind: {kind}")

    def inputs(self, samples):
        if self.kind == "ast":
            return self.extractor(samples, sampling_rate=16000, return_tensors="np")["input_values"]
        return maest_input(samples, self.preprocessor)

    @torch.inference_mode()
    def forward(self, samples):
        values = torch.from_numpy(np.asarray(self.inputs(samples), dtype=np.float32))
        output = self.model.audio_spectrogram_transformer(values, output_hidden_states=True)
        logits = self.model.classifier(output.pooler_output)
        return transformer_vector(output.hidden_states, output.last_hidden_state), logits[0].numpy()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size):
        x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    """PANNs CNN14 at 16 kHz: n_fft 512, hop 160, 64 mel bands 50-8000 Hz, 527 AudioSet classes."""

    def __init__(self):
        super().__init__()
        self.stft_real = nn.Parameter(torch.zeros(257, 1, 512), requires_grad=False)
        self.stft_imag = nn.Parameter(torch.zeros(257, 1, 512), requires_grad=False)
        self.mel = nn.Parameter(torch.zeros(257, 64), requires_grad=False)
        self.bn0 = nn.BatchNorm2d(64)
        widths = (1, 64, 128, 256, 512, 1024, 2048)
        self.blocks = nn.ModuleList(ConvBlock(a, b) for a, b in pairwise(widths))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)

    def forward(self, waveform):
        x = F.pad(waveform[:, None], (256, 256), mode="reflect")  # torchlibrosa: center, reflect
        real, imag = F.conv1d(x, self.stft_real, stride=160), F.conv1d(x, self.stft_imag, stride=160)
        power = (real ** 2 + imag ** 2).transpose(1, 2)[:, None]  # (batch, 1, time, 257)
        x = 10 * torch.log10(torch.clamp(power @ self.mel, min=1e-10))  # ref 1.0, amin 1e-10, no top_db
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for position, block in enumerate(self.blocks):
            x = block(x, (1, 1) if position == len(self.blocks) - 1 else (2, 2))
        x = x.mean(dim=3)
        x = x.max(dim=2).values + x.mean(dim=2)
        embedding = F.relu(self.fc1(x))
        return embedding, torch.sigmoid(self.fc_audioset(embedding))


def load_cnn14(path: Path) -> Cnn14:
    import codecs

    # The legacy checkpoint pickles numpy arrays: allow only these data-rebuilding globals.
    allowed = [(np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
               np.ndarray, np.dtype, (codecs.encode, "_codecs.encode"),
               *(type(np.dtype(name)) for name in ("int64", "float32", "float64"))]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    renamed = {}
    for key, value in state.items():
        if key.startswith("spec_augmenter"):
            continue
        key = (key.replace("spectrogram_extractor.stft.conv_real.weight", "stft_real")
               .replace("spectrogram_extractor.stft.conv_imag.weight", "stft_imag")
               .replace("logmel_extractor.melW", "mel"))
        for index in range(6):
            key = key.replace(f"conv_block{index + 1}.", f"blocks.{index}.")
        renamed[key] = value
    model = Cnn14()
    model.load_state_dict(renamed, strict=True)
    return model.eval()


class Cnn14Encoder:
    def __init__(self, path: Path):
        self.model = load_cnn14(path)

    @torch.inference_mode()
    def forward(self, samples):
        embedding, clipwise = self.model(torch.from_numpy(np.asarray(samples, dtype=np.float32))[None])
        return embedding[0].numpy().astype(np.float64), clipwise[0].numpy()
