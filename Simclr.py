
from copy import deepcopy
import torch
import torch.utils.data
import torch.nn as nn

from pl_bolts.models.self_supervised import SimCLR

class SimCLRModel(nn.Module):
    def __init__(self, weight_path='https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'):
        super().__init__()

        backbone = deepcopy(SimCLR.load_from_checkpoint(weight_path, strict=False).encoder)
        backbone.fc = nn.Identity()

        self.encoder = backbone

        self.freeze()  # freeze last block of resnet18
        self.inplanes = self.encoder.inplanes


    def freeze(self):
        c = 0
        l = 0
        num_layer = 0
        for _ in self.encoder._modules['layer4'].parameters():
            num_layer += 1
        for _ in self.encoder.parameters():
            l += 1
        for params in self.encoder.parameters():
            if c < l - num_layer - 2:
                params.requires_grad = False
            c += 1

    def forward(self, x):
        x = self.encoder(x)
        x = x[0]
        a, b = x.shape
        x = torch.reshape(x, (a, b, 1, 1))

        return x
    
    def named_children(self):
        return self.encoder.named_children()
