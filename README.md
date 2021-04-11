# Deep-Perception-Project

Lior Shifman and Haim Isakov

This is the final project in the Learning and Perception - Fall 2020- 896874 Course, Bar Ilan.

The project is about trying to implement the SimCLR model as a backbone to the Faster-RCNN model, and training it to see how well the model learn.

The dataset used in this project is the Pascal VOC 2012, dowoaded in the main.py file with the following command:
```
MyPascalDataset(root='MyVOC2012', year='2012', image_set='train', download=True, transforms=get_transform(train=True))
```

The folowing packages were used by us in this poject:

• csv

• os

• copy 

• numpy 

• torch

• PIL

• matplotlib 

• torchvision

• xml

• pytorchcv

• pl_bolts


# How To Run?
Just run the following command inside main.py:

```
run("my_fasterrcnn", momentum=momentum, lr=lr, gamma=gamma)
```

Notice that there are some extra options, for example:

```
run(model_name, batch_size=4, gpu="0", lr=0.005, gamma=0.5, momentum=0.9)
```
