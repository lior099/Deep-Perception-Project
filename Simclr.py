
from copy import deepcopy
import torch
import torch.utils.data
import torch.nn as nn

from pl_bolts.models.self_supervised import SimCLR

class SimCLRModel(nn.Module):
    """
    SimCLR model class. The encoder is a resnet18 that we freeze its last block. Then, we add an average pooling layer
    and a fully connected block that has 3 linear layers (input, hidden, output).
    """
    def __init__(self, weight_path='https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'):
        super().__init__()

        backbone = deepcopy(SimCLR.load_from_checkpoint(weight_path, strict=False).encoder)
        backbone.fc = nn.Identity()
        # modules = list(list(backbone.children())[0].children())[:-1]  # delete the last fc layer.
        # backbone = nn.Sequential(*modules)

        self.encoder = backbone

        self.freeze()  # freeze last block of resnet18
        self.inplanes = self.encoder.inplanes

        # self.avgpool = self.encoder.avgpool  # average pool layer
        # numft = self.encoder.fc.in_features
        # fully connected block
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=numft, out_features=256),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=100),
        #     nn.ReLU(),
        #     nn.Linear(in_features=100, out_features=20))

    def freeze(self):
        """
        Function that freezes the last block of the resnet18, which is the encoder of the SimCLR model.
        """
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
        """
        Pass the input through the encoder layer, average pooling layer and the fully connected block.
        """
        x = self.encoder(x)
        x = x[0]
        a, b = x.shape
        x = torch.reshape(x, (a, b, 1, 1))

        # x = self.avgpool(x)
        # x = self.fc(x)
        return x
    def named_children(self):
        return self.encoder.named_children()
