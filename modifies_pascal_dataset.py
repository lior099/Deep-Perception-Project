from torchvision.datasets import VOCDetection
import torch
from torchvision.models.detection import FasterRCNN
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import sys
from engine import train_one_epoch, evaluate
import utils
from PIL import Image
import xml.etree.ElementTree as ET

class MyPascalDataset(VOCDetection):

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())

        num_objs = len(target['annotation']['object'])
        boxes = []
        for obj in target['annotation']['object']:
            xmin = int(obj['bndbox']["xmin"])
            xmax = int(obj['bndbox']["xmax"])
            ymin = int(obj['bndbox']["ymin"])
            ymax = int(obj['bndbox']["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels_dict = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7,
                       'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14,
                       'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}
        labels = torch.tensor([0]*num_objs, dtype=torch.int64)
        for i in range(num_objs):
            label = labels_dict[target['annotation']['object'][i]['name']] + 1
            labels[i] = label
        image_id = torch.tensor([index])
        target_dict = {}
        target_dict["boxes"] = boxes
        target_dict["labels"] = labels
        target_dict["image_id"] = image_id
        target_dict["area"] = area
        target_dict["iscrowd"] = torch.tensor([False]*num_objs)

        if self.transforms is not None:
            img, target_dict = self.transforms(img, target_dict)

        return img, target_dict



