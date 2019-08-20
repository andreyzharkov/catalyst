import torch
import torch.nn as nn
import torch.nn.functional as F

from catalyst.dl import registry


from typing import Any, Mapping, Dict, List, Union
from collections import OrderedDict  # noqa F401

from torch import nn
from torch.utils.data import DataLoader  # noqa F401

from catalyst.dl.core import Runner, Callback
from catalyst.dl.experiment import SupervisedExperiment
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
from catalyst.dl.utils.torch import _Model, _Criterion, _Optimizer, _Scheduler


class YAERunner(Runner):
    # _default_experiment = SupervisedExperiment

    def __init__(
        self,
        model: nn.Module = None,
        device=None,
        input_key: tuple = ("images", "targets_a", "targets_b"),
        output_key: tuple = ("images_a", "images_b",
                             "L_reconstruction", "L_implicit", "L_explicit", "L_classification"),
        input_target_key: str = "targets",
        classification_weight = None
    ):
        """
        @TODO update docs
        Args:
            input_key: Key in batch dict mapping to model input
            output_key: Key in output dict model output will be stored under
        """
        super().__init__(model=model, device=device)
        self.input_key = input_key
        self.output_key = output_key
        # self.target_key = input_target_key

        self._classification_loss = nn.NLLLoss(weight=classification_weight)

        # self._process_input = self._process_input_none
        # self._process_output = self._process_output_none

        # if isinstance(self.input_key, str):
        #     self._process_input = self._process_input_str
        # elif isinstance(self.input_key, (list, tuple)):
        #     self._process_input = self._process_input_list
        # else:
        #     self._process_input = self._process_input_none
        #
        # if isinstance(output_key, str):
        #     self._process_output = self._process_output_str
        # elif isinstance(output_key, (list, tuple)):
        #     self._process_output = self._process_output_list
        # else:
        #     self._process_output = self._process_output_none

    def _batch2device(self, batch: Mapping[str, Any], device):
        batch = super()._batch2device(batch, device)
        assert len(batch) == len(self.input_key)
        return dict((k, v) for k, v in zip(self.input_key, batch))

    def predict_batch(self, batch: Mapping[str, Any]):
        images = batch["images"]
        targets_a = batch["targets_a"]
        targets_b = batch["targets_b"]

        enc = self.model.encoder
        dec = self.model.decoder
        #
        expl_a, impl_a = enc(images)

        images_a = dec(targets_a, impl_a)
        expl_aa, impl_aa = enc(images_a)

        images_b = dec(targets_b, impl_a)
        expl_ab, impl_ab = enc(images_b)

        impl_loss = ((impl_aa - impl_ab) ** 2).mean()
        return {
            'images_a': images_a,
            'images_b': images_b,
            'expl_a': expl_a,  # logits for targets_a
            'expl_b': expl_ab,  # logits for targets_b
            'impl_loss': impl_loss
        }


@registry.Criterion
class PredictionMeanLoss(nn.Module):
    def forward(self, x, y):
        return torch.mean(x)


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


@registry.Criterion
class ReconstructionLoss(nn.MSELoss):
    pass
