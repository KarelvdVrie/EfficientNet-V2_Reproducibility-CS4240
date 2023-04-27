
EfficientNetV2-S implementation in Pytorch using stages as in the paper. A reproducibility project for the Deep Learning course CS4240 at the TU Delft. 

# EfficientNet-V2_Reproducibility-CS4240
In Machine Learning the topic of reproducibility has become highly relevant, given the need for sustainable and reusable innovation. The following blog post and corresponding repository present our reproducibility research proced regarding the EfficientNet-V2 convolution network originally presented by Mingxing Tan and Quoc V Le in their work "_[EfficientNet V2: Smaller Models and Faster Training](https://paperswithcode.com/paper/efficientnetv2-smaller-models-and-faster)_". The novel approach presented in the work employs progressive learning to achieve spead up and efficient process. 

We focus our exploration in three main parts: (1) reprodusability of results shown in the paper using the ImageNetTE dataset; (2) hyperparameter sensitivity of the progressive learning elements; (3) transferability of performance on different datasets. Every component of our study is executed individually using the PyTorch implementation ([pytorch/vision](https://github.com/pytorch/vision)) combined with ClearML. In the next sections we outline the details of every step of our process starting with introduction of the paper, methodology, steps and finishing with conclusions and relevant links.

_Disclaimer:_ This project is created as part of the Deep Learning course CS4240 at the TU Delft.

# Authors
Three people contributed to the creation of this project:

- Karel van de Vrie [k.w.vandevrie@student.tudelft.nl] - PyTorch implementation enrichment with stages, reproducibility, ClearML implementation
- Nikoletta Nikolova [n.d.nikolova@student.tudelft.nl] - Hyperparameters sensitivity and analysis
- Anna-Maria Klianeva [] - Alternative datasets experiments and performance evaluation
 
# Table of Content

- [Relevant Links](#Relevant-links)
- [Introduction](#introduction)
- [Reproducibility](#reproducibility)
    - [Progressive Learning Stages](#progressive-learning-stages)
    - [Training on TE](#training-on-te)
    - [Results](#results)
- [Hyperparemeters](#hyperparameters)
    - [Epochs](#epochs)
    - [Dropout Limits](#dropout-limits)
    - [Image Size](#image-sizes)
- [Alternative Data](#ada)
- [ClearML](#clearml)
- [Conclusions](#conclusion)
- [Relevant Links](#relevant-links)



# Introduction

## What is EfficientNet?

## What is progressive learning?
Progressive Learning is a training method proposed by the authors of EfficientNetV2. It splits the training process in separate stages, each of which has a certain set of hyperparameters. The aim is to provide a start with reduced image size and low regularization parameters (such as dropout) and graduately scale them up with every next stage. This allows the training to be speed up and provide the network with different learning settings at every stage. ..

# Reproducibility
To be able to try and reproduce the results of the original paper we chose to use the official implementation of EfficientNetV2 from PyTorch. While this framework provided a detailed code structure and working examples, we noticed that it was missing complete implementation of the progressive learning process that the paper originally applies. The code was lacking the ability to add the different stages and change the parameters corersponding to them (see Table 6 from the [paper](https://arxiv.org/pdf/2104.00298.pdf)). 

## Progressive Learning Stages

## Training on TE

We train on a NVIDIA GPU 3090TX using the dataset [ImageNetTE](https://github.com/fastai/imagenette), which is significantly smaller than ImageNet and contains the 10 easily classified classes.


## Results

{% include_relative Graphs/ImageNetTE-EfficientNetV2-S.html %}


What can be observed is that compared to the original paper, the current implementation appears to be ...


While the paper notes an accuracy of 83.9% when trained on ImageNet and 84.9% when trained on ImageNet 21k, we appear to achieve an accuracy of 89.75%. The achieved results indicate that what the authors originally show in the paper appears to be credible. The raise in accuracy can be explained by the construct of the chosen dataset. 

# Hyperparameters
When it comes to hyperparameters, the most interesting components for investigation are the ones, which are part of the progressive learning process. More specifically, we look at epochs per stage, dropout limits and image sizes. 

## Epochs
Currently a lot of networks are trained by running trainings for many epochs, not directly focusing on optimizing the ... 

The structure of the EfficientNetV2 network uses the concept of stages to make the network faster and more 


- Original settings: [87] epochs per stage
- Test parameters: [20, 40, 60] epochs per stage

## Dropout Limits
To assess the sensitivity of the network to the dropout limits, 5 different tests are performed, where for each test the network is trained in 4 stages, each consisting of 20 epochs. This way we can assess how the different ranges of adaptive regularization influence the overall performance. 

- Original settings: min = [0.1], max = [0.3]
- Test parameters: min = [0.1, 0.2], max = [0.3, 0.5, 0.7]

What can be seen in the results is ... 



## Image Sizes
The paper claims, that "_the training can be further sped up by progressively increasing the image size during training time_". To evaluate this and the overall sensitivity of the network to the image size limits, 5 different tests are performed, where for each test the network is trained in 4 stages, each consisting of 20 epochs. As the progressive increase of image size is crucial to the claims of the paper we take special look at the speed and accuracy of the different trainings. 


- Original settings: min = [128], max = [300]
- Test parameters: min = [128, 300], max = [200, 300, 400]


The authors of the paper note that the progressive increase of image size can cause a drop in accuracy which, as can be seen in our results, is ...

# Alternative Datasets

## Pneumonia

## Ants and Bees

## Monkeys


# ClearML

# Conclusion

# Relevant Links

- Our reprodusability: https://github.com/KarelvdVrie/EfficientNet-V2_Reproducibility-CS4240
- Original paper: https://arxiv.org/pdf/2104.00298.pdf
- Papers with code: https://paperswithcode.com/paper/efficientnetv2-smaller-models-and-faster
- PyTorch implementation: https://github.com/pytorch/vision
