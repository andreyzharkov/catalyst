import torch
import torch.nn as nn
import torch.nn.functional as F

from catalyst.dl import registry

################################################
# criterions ###################################
################################################


@registry.Criterion
class PredictionMeanLoss(nn.Module):
    def forward(self, x, y):
        return torch.mean(x)


@registry.Criterion
class ReconstructionLoss(nn.MSELoss):
    pass


################################################
# modules ######################################
################################################


@registry.Module
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


@registry.Module
class Reshape(nn.Module):
    def __init__(self, non_batch_shape):
        super().__init__()
        self.non_batch_shape = non_batch_shape

    def forward(self, x):
        return torch.reshape(x, x.size()[:1] + self.non_batch_shape)


@registry.Module
class SimpleDecoder(nn.Module):
    def __init__(self, ch_out=1, res_out=(32, 32), n_classes=10, implicit_dim=10, ch_base=32, embedding_dim=10):
        super().__init__()
        assert res_out == (32, 32)
        self.embedding = nn.Embedding(num_embeddings=n_classes, embedding_dim=embedding_dim)
        dense_dim = 100
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + implicit_dim, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, ch_base * 4 * 4),
            nn.BatchNorm1d(ch_base * 4 * 4),
            nn.ReLU(),
            Reshape((ch_base, 4, 4)),
            # (dense_dim, 4, 4)
            nn.Upsample(size=(8, 8)),
            nn.Conv2d(ch_base, ch_base * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_base * 2),
            nn.ReLU(),

            nn.Upsample(size=(16, 16)),
            nn.Conv2d(ch_base * 2, ch_base * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_base * 4),
            nn.ReLU(),

            nn.Upsample(size=(32, 32)),
            nn.Conv2d(ch_base * 4, ch_out, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, y, x):
        return self.net(torch.cat((self.embedding(y), x), dim=1))


@registry.Module
class SimpleEncoder(nn.Module):
    def __init__(self, ch_in=1, n_classes=10, implicit_dim=10, image_resolution=(32, 32), ch_base=16):
        super().__init__()
        assert image_resolution == (32, 32)

        self.n_classes = n_classes

        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_base, kernel_size=3, stride=2, padding=1),
            # 16x16
            nn.BatchNorm2d(ch_base, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(ch_base, ch_base * 2, kernel_size=3, stride=2, padding=1),
            # 8x8
            nn.BatchNorm2d(ch_base * 2, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(ch_base * 2, ch_base * 4, kernel_size=3, stride=2, padding=1),
            # 4x4
            nn.BatchNorm2d(ch_base * 4, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(ch_base * 4, ch_base * 8, kernel_size=3, stride=1, padding=0),
            # 2x2
            nn.BatchNorm2d(ch_base * 8, momentum=0.1),
            nn.ReLU(),

            Flatten(),  # 2*2*ch_base*8
            nn.Linear(2 ** 2 * ch_base * 8, n_classes + implicit_dim)
        )

    def forward(self, x):
        x_encoded = self.net(x)
        x_explicit, x_implicit = x_encoded[:, :self.n_classes], x_encoded[:, self.n_classes:]
        return x_explicit, x_implicit


################################################
# models #######################################
################################################


@registry.Model
class YAE(nn.Module):
    def __init__(self, ch_in=1, n_classes=10, implicit_dim=10, image_resolution=(32, 32), ch_base=16):
        super().__init__()
        assert image_resolution == (32, 32)

        self.n_classes = n_classes

        self.encoder = SimpleEncoder(ch_in=ch_in, n_classes=n_classes, implicit_dim=implicit_dim,
                                     image_resolution=image_resolution, ch_base=ch_base)

        self.decoder = SimpleDecoder(ch_out=ch_in, res_out=image_resolution,
                                     n_classes=n_classes, implicit_dim=implicit_dim,
                                     embedding_dim=n_classes // 2)

    def forward(self, x):
        x_explicit, x_implicit = self.encoder(x)
        return self.decoder(torch.argmax(x_explicit, dim=-1), x_implicit)
