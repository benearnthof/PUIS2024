# ROBUST PROBABILISTIC UNSUPERVISED SEGMENTATION WITH UNCERTAINTY MODELING

![PU-Segnet](https://github.com/benearnthof/PUIS2024/blob/main/pusegnet_git.png?raw=true)

Unsupervised semantic segmentation aims to assign a semantic label to each pixel in an image, identifying the object or scene class without any supervision.
However, the task becomes particularly difficult due to factors like unclear or overlapping boundaries, intricate object textures, and the presence of multiple objects within the same region.
Traditional unsupervised models often suffer from class misalignment and poor spatial coherence, leading to fragmented and imprecise segmentation, often employing postprocessing with Conditional Random Fields (CRFs) to improve their results.
Additionally, deterministic models lack the ability to capture prediction uncertainty, making their outputs particularly prone to errors in ambiguous regions. 
To address these issues, we propose a probabilistic unsupervised semantic segmentation framework that enhances the robustness and accuracy of segmentation by refining predictions through uncertainty modeling and spatial smoothing techniques.
We also introduce a novel loss function that encourages the model to focus on learning similarities within pixels by leveraging feature information from pre-trained vision transformer backbones.
We also provide theoretical analyses of our proposed loss function, highlighting its favorable properties in relation to the optimization of our models.
Our method demonstrates superior accuracy and calibration, outperforming various baselines across multiple unsupervised semantic segmentation benchmarks including COCO, Potsdam, and Cityscapes. 
In conclusion, our framework offers a foundation for more reliable, uncertainty-aware segmentation models, advancing research in unsupervised semantic segmentation.


This repository consists of multiple parts: A STEGO directory that contains everything needed to pretrain custom STEGO models with DINOv2 backbones, an equivalent EAGLE repository that contains the code to pretrain custom EAGLE backbones on DINOv2 and a PUIS directory that utilizes these pretrained backbone models to train an SNGP segmentation head to perform probabilistic unsupervised image segmentation. 
Training & Evaluation scripts are in the pusegnet directory. The pusegnet directory also contains a DINO directory where we provide a branch of the official DINOv2 code with minor changes added to allow returning attention and feature maps in the way DINOv1 implemented it.
