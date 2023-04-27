# EfficientNet-V2_Reproducibility-CS4240 Code
All code was run within a Docker container, specifically the Nvidia Pytorch image: `nvcr.io/nvidia/pytorch:23.03-py3`, the only other package installed was `clearml==1.10.3` using pip. There are examples on the internet how to run a Docker container thus this will not be explained here.

All the code is selfcontained to enable reproducibility in and out of ClearML. This results in 3 different Jupyter notebooks:
 - `EfficientNetV2-S_ImageNetTE.ipynb` holds the code which uses the staged training as described by the original paper.
 - `EfficientNetV2-S_Custom-Datasets.ipynb` holds the code which trains on new datasets.
 - `EfficientNetV2-S_ImageNetTE - Hyper Parameter Optimizer.ipynb` holds the code to setup the ClearML hyper parameter optimizer task with which the different values for the dropout, image sizes, and epochs were tested. This code was also used to tune the tasks which train on new datasets.

The different sets of hyper parameters can be found in the respective Jupyter notebooks. For the `EfficientNetV2-S_Custom-Datasets.ipynb` the different parameters can be found in the main blog post / README, here you can also find the datasets used to train the network, this data will have to be downloaded before you can use the Jupyter notebook.