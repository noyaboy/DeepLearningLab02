from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn


class LDB(nn.Module):
    """Lightweight Dense Block.

    Features are fused through element-wise summation inside the block while the
    block exposes only one new feature volume (via concatenation) to avoid
    channel explosion.
    """

    def __init__(self, in_channels: int, t: float = 0.5, layers: int = 4) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if not 0.0 < t <= 1.0:
            raise ValueError("t must be in the range (0, 1]")
        if layers <= 0:
            raise ValueError("layers must be positive")

        growth_channels = max(16, int(round(in_channels * t)))
        self.growth_channels = growth_channels
        self.num_layers = layers

        self.initial = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_channels, kernel_size=3, padding=1, bias=False),
        )

        inner_layers = []
        for _ in range(max(0, layers - 1)):
            inner_layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(growth_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(growth_channels, growth_channels, kernel_size=3, padding=1, bias=False),
                )
            )
        self.layers = nn.ModuleList(inner_layers)

        self.out_channels = in_channels + growth_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning concatenated input and fused features."""
        fused = self.initial(x)
        for layer in self.layers:
            fused = fused + layer(fused)
        return torch.cat([x, fused], dim=1)


class TransitionLayer(nn.Module):
    """Channel compression + spatial downsampling between LDBs."""

    def __init__(self, in_channels: int, t: float = 0.5, dropout: float = 0.0) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if not 0.0 < t <= 1.0:
            raise ValueError("t must be in the range (0, 1]")

        out_channels = max(16, int(round(in_channels * t)))
        layers = [
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))
        self.transition = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transition(x)


class CDenseNet(nn.Module):
    """Simplified Compressed DenseNet for crowd counting on UCSD Pedestrian."""

    def __init__(
        self,
        n: int = 16,
        t: float = 0.5,
        num_classes: int = 3,
        initial_channels: int = 32,
        ldb_layers: int = 4,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.0,
        use_head_bn: bool = True,
    ) -> None:
        super().__init__()
        if n <= 0:
            raise ValueError("n must be positive")
        if initial_channels <= 0:
            raise ValueError("initial_channels must be positive")

        self.n = n
        self.t = t
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(1, initial_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
        )

        features: List[tuple[str, nn.Module]] = []
        channels = initial_channels
        for idx in range(n):
            ldb = LDB(channels, t=t, layers=ldb_layers)
            features.append((f"ldb{idx+1}", ldb))
            channels = ldb.out_channels

            transition = TransitionLayer(channels, t=t)
            features.append((f"transition{idx+1}", transition))
            channels = transition.out_channels

        self.features = nn.Sequential(OrderedDict(features))
        self.final_channels = channels

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        classifier_layers: List[nn.Module] = [
            nn.Linear(self.final_channels, head_hidden_dim),
        ]
        if use_head_bn:
            classifier_layers.append(nn.BatchNorm1d(head_hidden_dim))
        classifier_layers.append(nn.ReLU(inplace=True))
        if head_dropout > 0.0:
            classifier_layers.append(nn.Dropout(p=head_dropout))
        classifier_layers.append(nn.Linear(head_hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
