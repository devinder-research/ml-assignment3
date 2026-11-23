"""Model definitions: MLP, CNN, ConvAutoencoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Plain fully-connected classifier on flattened CIFAR10 images."""

    def __init__(self, hidden=(512, 256), dropout=0.0, num_classes=10):
        super().__init__()
        layers = []
        in_dim = 3 * 32 * 32
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)
        self.dropout_p = dropout

    def forward(self, x):
        return self.net(x.flatten(1))


class CNN(nn.Module):
    """Small VGG-style CNN for CIFAR10.

    conv(3->32,s1) -> conv(32->64,s2) -> conv(64->128,s2) -> conv(128->256,s2)
    -> fc(4*4*256 -> 256) -> fc(256 -> 10)
    Roughly parameter-matched to the ~1.7M-parameter MLP baseline.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 16x16
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 8x8
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def penultimate(self, x):
        """Features before the final linear layer (256-d)."""
        h = self.features(x)
        h = self.classifier[0](h)          # flatten
        h = self.classifier[1](h)          # fc 4096->256
        return F.relu(h, inplace=True)


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder with a flat bottleneck of size `bottleneck`.

    Encoder: conv(3->32,s2) -> conv(32->64,s2) -> conv(64->128,s2) -> fc(4*4*128->B)
    Decoder: fc(B->4*4*128) -> deconv(128->64,k4,s2) -> deconv(64->32,k4,s2)
             -> deconv(32->3,k4,s2) -> sigmoid
    """

    def __init__(self, bottleneck=256):
        super().__init__()
        self.bottleneck = bottleneck
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 8
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 4
            nn.Flatten(),                                                     # 2048
            nn.Linear(2048, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 2048), nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 8
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),   # 16
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),                                                     # 32
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summarize(model):
    n = count_parameters(model)
    print(f"{model.__class__.__name__}: {n:,} parameters")
    return n
