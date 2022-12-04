# Polyp-Segmentation
Download the Kvasir dataset here: https://datasets.simula.no/kvasir-seg/


## Metric used: 
Dice loss. Well suited for segmentation tasks, specifically on medical images, where there is a strong class imbalance between foreground and background. Considering the only two classes in this dataset are background and polyp, this metric seems like a good fit.

![image](https://user-images.githubusercontent.com/23001669/205491760-e14ff221-f801-4f8c-9a7c-632aecf6b153.png)
source:https://arxiv.org/pdf/2006.14822.pdf

## Metric target: .35

## Metric achieved: .4094 from hyperparameter experiment trial 16
![loss_plot](https://user-images.githubusercontent.com/23001669/205491637-f2e19c2b-44f0-4ced-89ad-2a9e4751fbbf.jpg)

