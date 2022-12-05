# Polyp-Segmentation
Download the Kvasir dataset here: https://datasets.simula.no/kvasir-seg/


## Metric used: 
Dice loss. Well suited for segmentation tasks, specifically on medical images, where there is a strong class imbalance between foreground and background. Considering the only two classes in this dataset are background and polyp, this metric seems like a good fit.

![image](https://user-images.githubusercontent.com/23001669/205491760-e14ff221-f801-4f8c-9a7c-632aecf6b153.png)
source:https://arxiv.org/pdf/2006.14822.pdf

## Metric target: .35

## Metric achieved: .4094 from hyperparameter experiment trial 16
![loss_plot](https://user-images.githubusercontent.com/23001669/205491637-f2e19c2b-44f0-4ced-89ad-2a9e4751fbbf.jpg)

## Time spent
Data exploration/manipulation/discovery: 5 hours  
Familiarizing myself with model/finding implementation: 5 hours  
Creating baseline models: 10 hours  
Encoder experiments: 7 hours  
Hyperparameter experiments: 10 hours  
Loss analysis: 1 hour  

## Change to original plan
Ditched stacked encoder-decoder network which I originally planned in the first assignment and used Linknet instead. 

## Pipeline
First, I trained a baseline model with UNet++ and a baseline model with Linknet. I decided to focus mainly on UNet++ after this because the validation loss was better and UNet++ was designed specifically for medical imaging. 

Then, I ran an experiment to determine the best backbone encoder. The results are saved in /encoder_logs. Mobile net performed slightly better than the rest and since it was by far the smallest backbone in terms of parameters, I chose this is as the backbone moving forward. I needed to take into consideration my limited computing resources on Google Colab and opt for the model which uses the least amount of memory.

Then I performed hyperparameter tuning. I created a spreadsheet of hyperparameter combos called hyperparameters_unetplusplus.csv for the segmentation model which I imported into the hyperparameter_experiments.ipynb script. I iterated over each row, passing the respective combo of hyperparameters to a newly instantiated UNet++ model, and saved the losses and some inferred test images in /hyperparameter_logs. After assessing the experiment in hp_tuning_loss_analysis.ipynb, I decided experiment 16's hyperparameters performed the best. An interesting takeaway from this experiment is that UNet++ trained with an encoder depth=5 dramatically outperformed encoder depths of 3 or 4.

For both experiments, the losses were analyzed along with a selection of images from the test set with high variance in polyp shape which were inferred by the model and analyzed visually.

Next I will work on deploying this model in the form of a web app. I plan on allowing the user to choose from a list of colonoscopy images which they would like to infer. Since the model is already performing well, I want to focus more on deployment since I have limited experience in this area. However, if I finish the web app early, I would like to continue hyperparameter tuning because I think there is room for improvement.
