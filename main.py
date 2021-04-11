import csv
import os
from copy import deepcopy

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import torchvision
from torchvision.datasets import VOCDetection
import xml.etree.ElementTree as ET
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import faster_rcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pytorchcv.model_provider import get_model as ptcv_get_model
from fasterrcnn import my_fasterrcnn_resnet50_fpn
from pl_bolts.models.self_supervised import SimCLR
from Simclr import SimCLRModel


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
        labels = torch.tensor([0] * num_objs, dtype=torch.int64)
        for i in range(num_objs):
            label = labels_dict[target['annotation']['object'][i]['name']] + 1
            labels[i] = label
        image_id = torch.tensor([index])
        target_dict = {}
        target_dict["boxes"] = boxes
        target_dict["labels"] = labels
        target_dict["image_id"] = image_id
        target_dict["area"] = area
        target_dict["iscrowd"] = torch.tensor([False] * num_objs)

        if self.transforms is not None:
            img, target_dict = self.transforms(img, target_dict)

        return img, target_dict




def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_object_detection_model(num_classes):
    # load a Faster-RCNN object detection model pre-trained on COCO
    model = my_fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def run(model_name, batch_size=4, gpu="0", lr=0.005, gamma=0.5, momentum=0.9):
    print("Running",model_name, "batch_size =",batch_size, "gpu =",gpu, "lr =",lr, "gamma =",gamma, "momentum =",momentum)
    name = "_".join(['lr', str(lr), "gamma", str(gamma), 'momentum', str(momentum)])
    device = torch.device('cuda:'+gpu) if torch.cuda.is_available() else torch.device('cpu')

    # get the model using our helper function
    num_classes = 21
    # model = my_fasterrcnn_resnet50_fpn()
    model = get_object_detection_model(num_classes)

    print("Starting download")
    train_voc_data = MyPascalDataset(root='MyVOC2012', year='2012', image_set='train', download=True,
                                     transforms=get_transform(train=True))

    test_voc_data = MyPascalDataset(root='MyVOC2012', year='2012', image_set='val', download=True,
                                    transforms=get_transform(train=False))
    print("Finished download")

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        train_voc_data, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        test_voc_data, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)





    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=momentum, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=gamma)

    # let's train it for 100 epochs
    num_epochs = 100
    acc_list = []
    loss_list = []
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        loss_list.append(dict(logger.meters))
        update_file(os.path.join('Results', 'loss_'+name+'.csv'), dict(logger.meters), epoch)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluator = evaluate(model, data_loader_test, device=device, print_freq=100)
        acc_list.append({"Accuracy": evaluator.coco_eval['bbox'].stats[0]})
        update_file(os.path.join('Results', 'acc_'+name+'.csv'), {"Accuracy": evaluator.coco_eval['bbox'].stats[0]}, epoch)


def update_file(path, row_dict, epoch):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if epoch == 0:
        with open(path, 'w', newline='') as file:
            wr = csv.writer(file)
            wr.writerow([''] + list(row_dict.keys()))
    with open(path, 'a', newline='') as file:
        wr = csv.writer(file)
        wr.writerow([epoch] + list(row_dict.values()))

def plot_graph(path, color, x_label, y_label, save_path=None, index=1, title=''):
    with open(path, 'r', newline='') as file:
        data = list(csv.reader(file))
        x = [int(row[0]) for row in data[1:]]
        if y_label in ['Accuracy', 'Learning Rate']:
            y = [float(row[index]) for row in data[1:]]
        else:
            y = [float(row[index].split(" ")[-1][1:-1]) for row in data[1:]]
    if y_label == 'Accuracy':
        y = [100*i for i in y]
    plt.plot(x, y, "-ok", color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle='--', which="both")
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()



if __name__ == '__main__':
    momentum = 0.9
    lr = 0.03
    gamma = 0.8
    run("my_fasterrcnn", momentum=momentum, lr=lr, gamma=gamma)

