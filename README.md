# PUIS2024
Production Ready Implementation of PU-Segnet with Custom backbones.

![PU-Segnet](https://github.com/benearnthof/PUIS2024/blob/main/pusegnet_git.png?raw=true)

This repository consists of multiple parts: A STEGO directory that contains everything needed to pretrain custom STEGO models with DINOv2 backbones, an equivalent EAGLE repository that contains the code to pretrain custom EAGLE backbones on DINOv2 and a PUIS directory that utilizes these pretrained backbone models to train an SNGP segmentation head to perform probabilistic unsupervised image segmentation. 
Training & Evaluation scripts are in the pusegnet directory. The pusegnet directory also contains a DINO directory where we provide a branch of the official DINOv2 code with minor changes added to allow returning attention and feature maps in the way DINOv1 implemented it.
