import streamlit as st
import os
from torchvision import models
from flask import Flask, jsonify, request
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import torch.nn as nn
from catalyst.callbacks import EarlyStoppingCallback
from catalyst.runners import SupervisedRunner
from catalyst.callbacks import  CheckpointCallback 
import segmentation_models_pytorch as smp

x_test_dir = 'test/test/images'
y_test_dir = 'test/test/masks'
ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['polyp', 'background']
ACTIVATION = 'sigmoid'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig('x',dpi=400)
    st.image('x.png')

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.Resize(576, 736, always_apply=True, p=1),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(576, 736)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

class Dataset(BaseDataset):
    """Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['polyp', 'background']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            single_file=False
    ):
        
        if single_file:
            self.ids = images_dir
            self.images_fps = os.path.join('test/test/images', self.ids)
            self.masks_fps = os.path.join('test/test/masks', self.ids)
        else:
            self.ids = os.listdir(images_dir)
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps, 0)
        mask[np.where(mask < 8)] = 0
        mask[np.where(mask > 8)] = 255
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

def model_infer(img_name):
    
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        classes=len(CLASSES), 
        activation=ACTIVATION,
        decoder_attention_type=None,
    )


    model.load_state_dict(torch.load('best.pth', map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()

    test_dataset = Dataset(
        img_name,
        img_name,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        single_file=True
    )

    test_dataloader = DataLoader(test_dataset)

    loaders = {"infer": test_dataloader}

    runner = SupervisedRunner()

    logits = []
    f = 0
    for prediction in runner.predict_loader(model=model, loader=loaders['infer'],cpu=True):
        if f < 3:
            logits.append(prediction['logits'])
            f = f + 1
        else:
            break

    threshold = 0.5
    break_at = 1

    for i, (input, output) in enumerate(zip(
            test_dataset, logits)):
        image, mask = input

        image_vis = image.transpose(1, 2, 0)
        gt_mask = mask[0].astype('uint8')
        pr_mask = (output[0].numpy() > threshold).astype('uint8')[0]
        i = i + 1
        if i >= break_at:
            break
    
    return image_vis, gt_mask, pr_mask

PAGE_TITLE = "Polyp Segmentation"

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def file_selector_ui():
    folder_path = './test/test/images'
    filename = file_selector(folder_path=folder_path)
    printname = list(filename)
    printname[filename.rfind('\\')] = '/'
    st.write('You selected`%s`' % ''.join(printname))
    return filename

def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.subheader("Get the file path")
    image_path = file_selector_ui()
    image_path = os.path.abspath(image_path)
    st.write('to infer`%s`' % image_path[image_path.rfind("\\") + 1:])
    to_infer = image_path[image_path.rfind("\\") + 1:]

    if os.path.isfile(image_path) is True:
        file_name = os.path.basename(image_path)
        _, file_extension = os.path.splitext(image_path)
        if file_extension == ".jpg":
            image_vis, gt_mask, pr_mask = model_infer(to_infer)
            visualize(
                image=image_vis, 
                ground_truth_mask=gt_mask, 
                predicted_mask=pr_mask
            )            

if __name__ == "__main__":
    main()