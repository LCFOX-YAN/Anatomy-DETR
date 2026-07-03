"""Torchvision ResNet backbone for stride-8/16/32 feature extraction."""

from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torchvision

from ..core import register
from .common import FrozenBatchNorm2d, freeze_batch_norm2d


def _resnet_weights(name: str, pretrained: bool):
    if not pretrained:
        return None
    lname = str(name).lower()
    try:
        if lname == 'resnet50':
            return torchvision.models.ResNet50_Weights.DEFAULT
        if lname == 'resnet101':
            return torchvision.models.ResNet101_Weights.DEFAULT
        if lname == 'resnet34':
            return torchvision.models.ResNet34_Weights.DEFAULT
        if lname == 'resnet18':
            return torchvision.models.ResNet18_Weights.DEFAULT
    except AttributeError:
        return None
    return None


@register()
class TorchvisionResNet(nn.Module):
    """Return C3/C4/C5-style features from a torchvision ResNet.

    ``return_idx`` uses the stage indices [0, 1, 2, 3] = [layer1, layer2, layer3, layer4].
    For FPN/HybridEncoder stride [8,16,32], use return_idx=[1,2,3], giving channels
    [512, 1024, 2048] for ResNet-50/101.
    """

    def __init__(self,
                 name: str = 'resnet50',
                 pretrained: bool = True,
                 return_idx=(1, 2, 3),
                 freeze_at: int = -1,
                 freeze_stem_only: bool = False,
                 freeze_norm: bool = False,
                 norm_layer: str = 'batchnorm',
                 local_weight_path: str = ''):
        super().__init__()
        self.name = str(name).lower()
        self.return_idx = tuple(int(i) for i in return_idx)

        norm = FrozenBatchNorm2d if str(norm_layer).lower() in ('frozenbatchnorm2d', 'frozen_bn', 'frozen') else nn.BatchNorm2d
        builder = getattr(torchvision.models, self.name, None)
        if builder is None:
            raise ValueError(f'Unsupported torchvision ResNet backbone: {name}')

        weights = _resnet_weights(self.name, pretrained and not local_weight_path)
        try:
            self.body = builder(weights=weights, norm_layer=norm)
        except TypeError:
            self.body = builder(pretrained=bool(pretrained and not local_weight_path), norm_layer=norm)

        if local_weight_path:
            if not os.path.exists(local_weight_path):
                raise FileNotFoundError(f'local_weight_path does not exist: {local_weight_path}')
            state = torch.load(local_weight_path, map_location='cpu')
            if isinstance(state, dict) and 'model' in state:
                state = state['model']
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            self.body.load_state_dict(state, strict=False)
            
        if hasattr(self.body, 'fc'):
            self.body.fc = nn.Identity()

        stage_channels = {
            'resnet18': (64, 128, 256, 512),
            'resnet34': (64, 128, 256, 512),
            'resnet50': (256, 512, 1024, 2048),
            'resnet101': (256, 512, 1024, 2048),
            'resnet152': (256, 512, 1024, 2048),
        }
        self._out_channels = [stage_channels.get(self.name, stage_channels['resnet50'])[i] for i in self.return_idx]
        self._out_strides = [(4, 8, 16, 32)[i] for i in self.return_idx]

        if freeze_norm:
            self.body = freeze_batch_norm2d(self.body)
            for m in self.body.modules():
                if isinstance(m, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                    m.eval()
                    for p in m.parameters(recurse=False):
                        p.requires_grad = False

        self._freeze_stages(freeze_at, freeze_stem_only)

    def _freeze_module(self, module: nn.Module):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def _freeze_stages(self, freeze_at: int, freeze_stem_only: bool):
        # freeze_at >= 0 freezes the stem.  freeze_at >= 1 additionally freezes layer1, etc.
        if freeze_stem_only or int(freeze_at) >= 0:
            self._freeze_module(self.body.conv1)
            self._freeze_module(self.body.bn1)
        for idx, layer in enumerate([self.body.layer1, self.body.layer2, self.body.layer3, self.body.layer4], start=1):
            if int(freeze_at) >= idx:
                self._freeze_module(layer)

    def forward(self, x):
        x = self.body.conv1(x)
        x = self.body.bn1(x)
        x = self.body.relu(x)
        x = self.body.maxpool(x)

        outs = []
        for idx, layer in enumerate([self.body.layer1, self.body.layer2, self.body.layer3, self.body.layer4]):
            x = layer(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
