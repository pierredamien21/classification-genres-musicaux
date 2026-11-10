"""V2 reinforced CNN: the V1 SmallAudioCNN layout with batch normalization, plus SpecAugment.

models/cnn.py and training/cnn.py are pinned by the V1 lock, so the V2 additions live here.
"""

import torch
from torch import nn


class ReinforcedAudioCNN(nn.Module):
    def __init__(self, classes=8):
        super().__init__()
        blocks = []
        incoming = 1
        for position, outgoing in enumerate((16, 32, 64)):
            blocks.extend((nn.Conv2d(incoming, outgoing, 3, stride=2 if position == 0 else 1, padding=1,
                                     bias=False), nn.BatchNorm2d(outgoing), nn.ReLU(), nn.MaxPool2d(2)))
            incoming = outgoing
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, classes)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


def spec_augment(batch, config, generator):
    """Zero random frequency and time bands (width 0..max) per example; returns a new tensor."""
    out = batch.clone()
    _, _, mels, frames = out.shape
    for example in out:
        for count, width, size, axis in ((config["freq_masks"], config["freq_width"], mels, 1),
                                         (config["time_masks"], config["time_width"], frames, 2)):
            for _ in range(count):
                w = int(torch.randint(0, width + 1, (1,), generator=generator))
                start = int(torch.randint(0, size - w + 1, (1,), generator=generator))
                example.narrow(axis, start, w).zero_()
    return out
