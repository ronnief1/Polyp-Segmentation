# Polyp-Segmentation
Download the Kvasir dataset here: https://datasets.simula.no/kvasir-seg/


## Metric used: 
Dice loss. Well suited for segmentation tasks, specifically on medical images, where there is a strong class imbalance between foreground and background. Considering the only two classes in this dataset are background and polyp, this metric seems like a good fit.

![image](https://user-images.githubusercontent.com/23001669/205491168-15496a83-04f4-439b-a569-2f42cd007d8e.png)
source: https://stats.stackexchange.com/questions/500628/what-happens-when-y-true-is-all-0-in-dice-loss

## Metric target: .4

## Metric achieved: .4094 from hyperparameter experiment trial 16
![loss_plot](https://user-images.githubusercontent.com/23001669/205491637-f2e19c2b-44f0-4ced-89ad-2a9e4751fbbf.jpg)

