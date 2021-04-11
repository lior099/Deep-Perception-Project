
import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import MultiScaleRoIAlign
import torchvision.models.detection as detection

from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.utils import load_state_dict_from_url



from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers, mobilenet_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN

import warnings
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.ops import misc as misc_nn_ops
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import mobilenet
from torchvision.models import resnet
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from Simclr import SimCLRModel



def my_fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=21, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    backbone = SimCLRModel()
    backbone = my_resnet_fpn_backbone(backbone, pretrained_backbone, trainable_layers=trainable_backbone_layers)
    out_channels = 256
    backbone.out_channels = out_channels
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model

def my_resnet_fpn_backbone(
    backbone,
    pretrained,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None
):



    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
