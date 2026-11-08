"""Small CNN trained from scratch on native-resolution log-Mel windows."""

import torch
from torch import nn


class SmallAudioCNN(nn.Module):
    def __init__(self, classes=8):
        super().__init__()
        blocks = []
        incoming = 1
        for position, outgoing in enumerate((16, 32, 64)):
            blocks.extend((nn.Conv2d(incoming, outgoing, 3, stride=2 if position == 0 else 1,
                                     padding=1), nn.ReLU(), nn.MaxPool2d(2)))
            incoming = outgoing
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, classes)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


def normalize_logmel(x, db_min=-80.0, db_max=20.0):
    if db_max <= db_min:
        raise ValueError("Expected increasing dB bounds")
    return (torch.clamp((x - db_min) / (db_max - db_min), 0, 1) - 0.5) / 0.5
