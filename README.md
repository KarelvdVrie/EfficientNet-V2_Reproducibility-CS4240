
EfficientNetV2-S implementation in Pytorch using stages as in the paper. A reproducibility project for the Deep Learning course CS4240 at the TU Delft. 

# EfficientNet-V2_Reproducibility-CS4240
In Machine Learning the topic of reproducibility has become highly relevant, given the need for sustainable and reusable innovation. The following blog post and corresponding repository present our reproducibility research proced regarding the EfficientNet-V2 convolution network originally presented by Mingxing Tan and Quoc V Le in their work "_[EfficientNet V2: Smaller Models and Faster Training](https://paperswithcode.com/paper/efficientnetv2-smaller-models-and-faster)_". The novel approach presented in the work employs progressive learning to achieve spead up and efficient process. 

We focus our exploration in three main parts: (1) reprodusability of results shown in the paper using the ImageNetTE dataset; (2) hyperparameter sensitivity of the progressive learning elements; (3) transferability of performance on different datasets. Every component of our study is executed individually using the PyTorch implementation ([pytorch/vision](https://github.com/pytorch/vision)) combined with ClearML. In the next sections we outline the details of every step of our process starting with introduction of the paper, methodology, steps and finishing with conclusions and relevant links.

_Disclaimer:_ This project is created as part of the Deep Learning course CS4240 at the TU Delft.

# Authors
Three people contributed to the creation of this project:

- Karel van de Vrie [k.w.vandevrie@student.tudelft.nl] - PyTorch implementation enrichment with stages, reproducibility, ClearML implementation
- Nikoletta Nikolova [n.d.nikolova@student.tudelft.nl] - Hyperparameters sensitivity and analysis
- Anna-Maria Klianeva [a.v.klianeva@student.tudelft.nl] - Alternative datasets experiments and performance evaluation
 
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




What can be observed is that compared to the original paper, the current implementation appears to be ...
{% include_relative Graphs/ImageNetTE-EfficientNetV2-S.html %}

While the paper notes an accuracy of 83.9% when trained on ImageNet and 84.9% when trained on ImageNet 21k, we appear to achieve an accuracy of 89.75%. The achieved results indicate that what the authors originally show in the paper appears to be credible. The raise in accuracy can be explained by the construct of the chosen dataset. 

# Hyperparameters
When it comes to hyperparameters, the most interesting components for investigation are the ones, which are part of the progressive learning process. More specifically, we look at epochs per stage, dropout limits and image sizes. 

## Epochs
Currently a lot of networks are trained by running trainings for many epochs, not directly focusing on optimizing the ... 

The structure of the EfficientNetV2 network uses the concept of stages to make the network faster and more 


- Original settings: [87] epochs per stage
- Test parameters: [20, 40, 60] epochs per stage

{% include_relative Graphs/Epochs.html %}

## Dropout Limits
To assess the sensitivity of the network to the dropout limits, 5 different tests are performed, where for each test the network is trained in 4 stages, each consisting of 20 epochs. This way we can assess how the different ranges of adaptive regularization influence the overall performance. 

- Original settings: min = [0.1], max = [0.3]
- Test parameters: min = [0.1, 0.2], max = [0.3, 0.5, 0.7]

What can be seen in the results is ... 

{% include_relative Graphs/Dropout_parallel.html %}


## Image Sizes
The paper claims, that "_the training can be further sped up by progressively increasing the image size during training time_". To evaluate this and the overall sensitivity of the network to the image size limits, 5 different tests are performed, where for each test the network is trained in 4 stages, each consisting of 20 epochs. As the progressive increase of image size is crucial to the claims of the paper we take special look at the speed and accuracy of the different trainings. 


- Original settings: min = [128], max = [300]
- Test parameters: min = [128, 300], max = [200, 300, 400]


The authors of the paper note that the progressive increase of image size can cause a drop in accuracy which, as can be seen in our results, is ...
{% include_relative Graphs/Image_sizes.html %}


# Alternative Datasets
For the project, the *Efficientnet_v2_s* is used with three different datasets. The *Efficientnet_v2_s* model was imported from torchvision.models and the pretrained weights are used unless when specifically mentioned otherwise. This is done to evaluate the performance of the model on a new data. 

## Pneumonia
The use of Deep Learning (DL) in medicine is becoming increasingly popular [1]. One impornant application is the use of DL for disease's detection. Thus, the dataset Chest X-Ray Images (Pneumonia) [2] was used for this project. This dataset X-ray images of normal chest, as well as X-ray images of chest with bacterial and viral pnemonia. The images are split in two classes: Normal and Pnemonia.

*Efficientnet_v2_s* with different hyperparameter sets are tested. The set of parameters is shown in the table below.
|  Parameter | Value(s) | 
| ------------- | ------------- |
| Learning rate  | 0.001-0.03  |
| Batch size  | 10, 20, 30  |
| Epochs | 10, 15, 20, 30 |

The validation accuracy for different combination of hyperparameters is shown in the table below:
|task id                                                                                                                                                                |objective|iteration|General/batch_size|General/epochs|General/learning_rate|status   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|---------|------------------|--------------|---------------------|---------|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/d5762cb3efb4416ea1508632b3885860/output/log"> d5762cb3efb4416ea1508632b3885860 </a>|0.75     |29       |30                |30            |0.021                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/6a90e51d77e345b49c1547b5c74f8b94/output/log"> 6a90e51d77e345b49c1547b5c74f8b94 </a>|0.6875   |29       |20                |30            |0.025                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/d0f1f0cf492a4623a19d9e65f38e1d5d/output/log"> d0f1f0cf492a4623a19d9e65f38e1d5d </a>|0.6875   |29       |20                |30            |0.003                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/1ab6bf5d04134e5182dec2165c4048e3/output/log"> 1ab6bf5d04134e5182dec2165c4048e3 </a>|0.6875   |9        |20                |10            |0.003                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/7370112ef195407ab17ead3d652423ca/output/log"> 7370112ef195407ab17ead3d652423ca </a>|0.6875   |14       |10                |15            |0.021                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/6909452ee5ee422a809d6dad89fb9292/output/log"> 6909452ee5ee422a809d6dad89fb9292 </a>|0.6875   |14       |10                |15            |0.027000000000000003 |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/a7530da366b64af38a0f3095b96df214/output/log"> a7530da366b64af38a0f3095b96df214 </a>|0.6875   |19       |30                |20            |0.017                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/9cd3f374b7d746708e1c4ad8311c7f83/output/log"> 9cd3f374b7d746708e1c4ad8311c7f83 </a>|0.6875   |19       |10                |20            |0.001                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/80f5b4bcb1d0467d8bb9ced74afe9a0f/output/log"> 80f5b4bcb1d0467d8bb9ced74afe9a0f </a>|0.6875   |9        |20                |10            |0.013000000000000001 |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/434c0ebdf1344fb8aef7391a974d0baf/output/log"> 434c0ebdf1344fb8aef7391a974d0baf </a>|0.6875   |9        |20                |10            |0.011                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/a7244678daaa40399fd5785231f107c6/output/log"> a7244678daaa40399fd5785231f107c6 </a>|0.6875   |13       |10                |15            |0.023                |stopped  |
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/79846bb4e68448a7a04915f1e971246e/output/log"> 79846bb4e68448a7a04915f1e971246e </a>|0.625    |19       |30                |20            |0.019000000000000003 |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/b3ae31aeb6034747b7191d5c3662083e/output/log"> b3ae31aeb6034747b7191d5c3662083e </a>|0.625    |14       |30                |15            |0.019000000000000003 |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/d9579f401d594a5b8a918239b863f6a0/output/log"> d9579f401d594a5b8a918239b863f6a0 </a>|0.625    |19       |20                |20            |0.021                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/a961f349e67a4ec996d55dd5eeb5e5ec/output/log"> a961f349e67a4ec996d55dd5eeb5e5ec </a>|0.625    |14       |30                |15            |0.013000000000000001 |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/a7ca223bf77d4d75b23458e6dbcfd734/output/log"> a7ca223bf77d4d75b23458e6dbcfd734 </a>|0.625    |19       |30                |20            |0.025                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/274bc3feb468404080e2a54bf3456b22/output/log"> 274bc3feb468404080e2a54bf3456b22 </a>|0.625    |27       |10                |30            |0.019000000000000003 |stopped  |
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/605ffbbf65294a7a8515f979c90c44ba/output/log"> 605ffbbf65294a7a8515f979c90c44ba </a>|0.625    |19       |10                |20            |0.007                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/66d82fda71a7444dacaab9b6e1615d89/output/log"> 66d82fda71a7444dacaab9b6e1615d89 </a>|0.625    |14       |20                |15            |0.019000000000000003 |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/8080292921434898b9d4d6ce379a43b3/output/log"> 8080292921434898b9d4d6ce379a43b3 </a>|0.625    |19       |10                |20            |0.017                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/34a04acb81fe4e3b9289f926369c7289/output/log"> 34a04acb81fe4e3b9289f926369c7289 </a>|0.625    |14       |30                |15            |0.017                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/401387b3ba4d420bbace0cb0fd7a6f27/output/log"> 401387b3ba4d420bbace0cb0fd7a6f27 </a>|0.625    |9        |30                |10            |0.025                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/0bb18c954deb4e419391e296753d1a8a/output/log"> 0bb18c954deb4e419391e296753d1a8a </a>|0.625    |29       |20                |30            |0.023                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/67ab2b4062ce4b8681a83762707c14d8/output/log"> 67ab2b4062ce4b8681a83762707c14d8 </a>|0.625    |29       |30                |30            |0.001                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/457030dfcc7f44fea2a755d2a4dc1183/output/log"> 457030dfcc7f44fea2a755d2a4dc1183 </a>|0.625    |19       |30                |20            |0.023                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/3d20cd6041504d948c7f977f14a019bf/output/log"> 3d20cd6041504d948c7f977f14a019bf </a>|0.625    |14       |10                |15            |0.009000000000000001 |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/74ba88ed6502440da292ca3732ffba16/output/log"> 74ba88ed6502440da292ca3732ffba16 </a>|0.625    |19       |30                |20            |0.015                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/862208a4a5694008b636ca7b9a00567d/output/log"> 862208a4a5694008b636ca7b9a00567d </a>|0.625    |14       |30                |15            |0.001                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/fc6c037c5f9a4ec3806a46866dd2bb68/output/log"> fc6c037c5f9a4ec3806a46866dd2bb68 </a>|0.625    |14       |20                |15            |0.013000000000000001 |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/f78358b6b0ca4124820a5a4f74c428f8/output/log"> f78358b6b0ca4124820a5a4f74c428f8 </a>|0.5625   |19       |30                |20            |0.003                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/7542ba62973d4f0b83335d6c4e26ed48/output/log"> 7542ba62973d4f0b83335d6c4e26ed48 </a>|0.5625   |19       |20                |20            |0.013000000000000001 |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/380e4306ec024b30884f952fb4f44f51/output/log"> 380e4306ec024b30884f952fb4f44f51 </a>|0.5625   |19       |10                |20            |0.021                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/b590c431d1864375a7892a35bc2b9b0c/output/log"> b590c431d1864375a7892a35bc2b9b0c </a>|0.5625   |9        |10                |10            |0.017                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/fe608b3f81de4662bd50b7d6a818fe60/output/log"> fe608b3f81de4662bd50b7d6a818fe60 </a>|0.5625   |19       |20                |20            |0.007                |completed|
|<a href="https://app.clear.ml/projects/69c062da03f14007bcd348b3012b86d0/experiments/b9f901c9a3cf4037ae170e8e07977e26/output/log"> b9f901c9a3cf4037ae170e8e07977e26 </a>|0.5625   |19       |30                |20            |0.001                |completed|


The best performance of *Efficientnet_v2_s* with the dataset lead to accuracy of 75% on validation data and 90% on training data. This is somewhat lower than the results of Aakashnain (2018) where validation accuracy of 82.6% was reached using Depthwise Convolution and the results of Madz2000 (2020) where 87.5% accuracy was achieved using Convolutional Neural network. 

## Ants and Bees
For the project, a very small dataset was also used, the Hymenoptera dataset [5]. This dataset only contains 398 images in two classes: ants and bees. For this dataset different hypermaraters are tested and shown in the table below: 
|  Parameter | Value(s) | 
| ------------- | ------------- |
| Learning rate  | 0.006-0.02  |
| Batch size  | 10, 20, 30  |
| Epochs | 50, 75, 100 |

Futhermore, the effect of retraining the weights of the model was evaluated for those hyperparameters. 

As it could be seen by the results above, the best validation accuracy for the given hyperparameters with *Efficientnet_v2_s* for the dataset is 75% when the weight are retrained versus 69.9 % when the pretrained weights are used. This shows that training the weights leads to better results when working with this dataset. It should be noted that best validation performance of 94% was achieved using ResNet-18 with pretrained weights [6].

## Monkeys
The 10 Monkey Species dataset [7] is also used in the project. The dataset containts 1400 images of 10 classes: mantled howler,patas monkey, bald uakari, japanese macaque, pygmy marmoset, white headed capuchin, silvery marmoset, common squirrel monkey, black headed night monkey, and nilgiri langur. The set of parameters is shown in the table below.
|  Parameter | Value(s) | 
| ------------- | ------------- |
| Learning rate  | 0.0001-0.1  |
| Batch size  | 10, 20, 30 |
| Epochs | 15, 20, 30, 40, 50 |

The best validation accuracy on this dataset is the lowest, 57.3%.

# ClearML

# Conclusion

# References
[1] Egger, J., Gsaxner, C., Pepe, A. and Li, J. (2022). Medical deep learning—A systematic meta-review. Computer Methods and Programs in Biomedicine, [online] 221, pp.106874–106874. doi:https://doi.org/10.1016/j.cmpb.2022.106874.

[2] Mooney, P. (2018). Chest X-Ray Images (Pneumonia). [online] Kaggle.com. Available at: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia [Accessed 26 Apr. 2023].

[3] Aakashnain (2018). Beating everything with Depthwise Convolution. [online] Kaggle.com. Available at: https://www.kaggle.com/code/aakashnain/beating-everything-with-depthwise-convolution [Accessed 26 Apr. 2023].

[4] Madz2000 (2020). Pneumonia Detection using CNN(92.6% Accuracy). [online] Kaggle.com. Available at: https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy [Accessed 26 Apr. 2023].

‌[5] Tensorflow Notebooks (2022). Hymenoptera dataset. [online] Kaggle.com. Available at: https://www.kaggle.com/datasets/tensorflownotebooks/hymenoptera-dataset [Accessed 26 Apr. 2023].

[6] Tensorflow Notebooks (2022). Transfer Learning for Computer Vision Tutorial. [online] Kaggle.com. Available at: https://www.kaggle.com/code/tensorflownotebooks/transfer-learning-for-computer-vision-tutorial [Accessed 26 Apr. 2023].

‌[7] Mario (2018). 10 Monkey Species. [online] Kaggle.com. Available at: https://www.kaggle.com/datasets/slothkong/10-monkey-species [Accessed 26 Apr. 2023].

‌
‌

‌
# Relevant Links

- Our reprodusability: https://github.com/KarelvdVrie/EfficientNet-V2_Reproducibility-CS4240
- Original paper: https://arxiv.org/pdf/2104.00298.pdf
- Papers with code: https://paperswithcode.com/paper/efficientnetv2-smaller-models-and-faster
- PyTorch implementation: https://github.com/pytorch/vision
