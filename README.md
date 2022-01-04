# Reproducibility of Contrastive Clustering (CC)

This is the code for reproducing the paper “Contrastive Clustering” (AAAI 2021)

# Dependency

- python>=3.7
- pytorch>=1.6.0
- torchvision>=0.8.1
- munkres>=1.1.4
- numpy>=1.19.2
- opencv-python>=4.4.0.46
- pyyaml>=5.3.1
- scikit-learn>=0.23.2

# Usage

# Dataset

This repository contains datasets:CIFAR-10, CIFAR-100. 
To download ImageNet-10, use this link: https://www.kaggle.com/liusha249/imagenet10
And drag the ImageNet-10 dataset under "datasets" directory to use.

## Configuration

There is a configuration file “config/config.yaml”, where one can edit both the training and test options, and this is the only place that needs to be changed to obtain reproduced result.

## 3.1.1 Figure 2

To obtain figure 2 in 3.1.1, in config.yaml, set the start_epoch=0, epoch=100, and dataset=“ImageNet-10”, and run the train.py. After it finishes, run the fig3_generator and you will get figure2.

## 3.1.2 Table 1

To obtain Table 1, in config,yaml, set the dataset to be “CIFAR-10", start-epoch=1000, and run the cluster.py file, you should get the evaluation metrics for CC on CIFAR-10. Thereafter, set dataset=“CIFAR-100” and run the cluster.py and get the evaluation metric for CC on CIFAR-100.

For K-mean result, simply run the main.py file will show the result of K-mean on two different datasets.

## 3.1.3 Figure 3, Figure 4, Figure 5

In config.yaml, set the start_epoch=0, epoch=100, and dataset=“ImageNet-10” and run train/.py. Then Figure 3,4,5 can be obtained by simply running the fig4_generator.py.

## 3.2 Figure 6

Set config.yaml to be dataset="CIFAR-10", start_epoch=0, epoch=20, and run the train.py. After it finishes, run the fig3_generator.py to obtain the Figure 6

## 3.3 Table 2

First step, set the config file to be dataset="ImageNet-10", start_epoch=0, epoch=20

for ICH+CCH: change the 30th line's code of trian.py to be "loss = loss_instance + loss_cluster", and run train.py and cluster.py one by one

for ICH only: change the 30th line's code of trian.py to be "loss = loss_instance", and run train.py and cluster.py one by one
      
for CCH only: change the 30th line's code of trian.py to be "loss = loss_cluster", and run train.py and cluster.py one by one

## 3.3 Table 3

First step, set the config file to be dataset="ImageNet-10", start_epoch=0, epoch=20
 
for average(ICH,CCH): change the 30th line's code of trian.py to be "loss = (loss_instance + loss_cluster)/2", and run train.py and cluster.py one by one

for max(ICH,CCH): change the 30th line's code of trian.py to be "loss = torch.max(loss_instance, loss_cluster)", and run train.py and cluster.py one by one 

for min(ICH,CCH): change the 30th line's code of trian.py to be "loss = torch.min(loss_instance, loss_cluster)", and run train.py and cluster.py one by one            

