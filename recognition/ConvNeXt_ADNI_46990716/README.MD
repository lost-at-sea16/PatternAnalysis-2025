# Classification of Alzheimer's Disease using ConvNeXt

---

Student Number: 46990716
Student Name: Sophia Gleeson

## Table of contents

- [Classification of Alzheimer's Disease using ConvNeXt](#classification-of-alzheimers-disease-using-convnext)
  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
  - [ConvNeXt Overview](#convnext-overview)
    - [Basic Building Block of ConvNeXt](#basic-building-block-of-convnext)
    - [ConvNeXt Architecture](#convnext-architecture)
    - [ConvNeXt Small](#convnext-small)
  - [Dataset](#dataset)
    - [Dataset Preparation \& Segmentation](#dataset-preparation--segmentation)
  - [Model Training and Experimental Adjustments](#model-training-and-experimental-adjustments)
  - [Model Fine-Tuning and Experimental Adjustments](#model-fine-tuning-and-experimental-adjustments)
    - [The ImageNet to ADNI Gap](#the-imagenet-to-adni-gap)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Results and Analysis](#results-and-analysis)
    - [Analysis of accuracy on a patient level](#analysis-of-accuracy-on-a-patient-level)
  - [Dependencies and Usage](#dependencies-and-usage)
  - [Future Steps](#future-steps)
  - [References](#references)

## Introduction

Alzheimer's disease is an irreversible neurodegenerative disease caused by damage and deterioration of brain tissue, affects a person’s cognitive functions[^1]. Given that Alzheimer's disease affects millions of people worldwide, the ability to accurately and effectively diagnose the disease is critical for earlier diagnosis, treatment planning and efficiency and patient care.

Though brain imaging is only one of the critical diagnostic steps, it is important to be able to identify even subtle changes in the brain.

This project aims to perform binary classification of 2D brain MRIs from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset between subjects with Alzheimer's disease (AD) and subjects with Normal Cognition (NC). Using one of the latest vision models, ConvNeXt, I aim to achieve a minimum accuracy of 0.8 on the test set.

## ConvNeXt Overview

ConvNeXt was introduced in the paper *A ConvNet for the 2020s*[^2] as a convolutional neural network (CNN) modernised toward the design of a vision transformer (ViT). Through implementing aspects of ViTs architecture and using similar techniques used to train ViTs, Z. Liu et. all, kept the simplicity and efficiency of CNNs while achieving state of the art performance on image recognition tasks.

### Basic Building Block of ConvNeXt

The ConvNeXt Block is composed of a depthwise convolution, layer normalisation, pointwise convolutions and GELU activation.

The block starts with a 7x7 depthwise convolution, a special case of grouped convolutions, where the number of groups is equal to the number of channels. It operates on a per-channel basis, mixing information only in the spatial dimension and not across channels. The 7x7 kernel size is directly inspired by the large receptive fields of attention models in swim transformers. The features are then normalized using Layer Normalization, a standard in ViTs, instead of Batch Normalization. The final stage is channel mixing, which mimics ViTs MLP and is performer by two 1x1 pointwise convolutions. The first 1x1 convolution expands the number of channels, then the GELU activation function is applied, and the second 1x1 convolution projects the channels back down to the input dimension. These pointwise 1x1 convolutions create the inverted bottleneck pattern seen in transformers, which helps avoid information loss.

![Basic Block](images\Block_designs.png)
*Figure 1: Comparison between the basic blocks of Swim transformers, ResNet and ConvNeXt*

### ConvNeXt Architecture

![Architecture](images\Architecture_specifications.png)
*Figure 2: Comparison between the architectures of Swim transformers, ResNet and ConvNeXt*

First, the input image is passed through a “patchify stem” of 4x4, stride 4 non-overlapping convolution layer. This immediately downsamples the image to be 4 times smaller and embeds it into a higher channel dimension.

ConvNeXt then consists of a similar four-stage structure as ResNet-50, while the number of blocks per stage was adjusted to match the stage compute ratio of Swim Transformers. The number of blocks per stage depends on the ConvNeXt model.

![ConvNeXt Sizes](images\ConvNeXt_sizes.png)
*Figure 3: ConvNeXt block ratios*

Between the groups of ConvNeXt blocks and after the last one, downsampling layers reduce the spatial dimensions while increasing the number of feature channels, allowing the model to focus on higher-level patterns.

The final feature map is passed through the global average pooling (GAP) layer, which performs a dimensionality reduction while retaining the overall data features, and then passed to the fully connected layer for classification.

### ConvNeXt Small

This project uses ConvNeXt-small.

![ConvNeXt Small](images\Convnext_small.png)
*Figure 4: ConvNeXt-small Architecture, modified from [^3]*

## Dataset

The ADNI dataset used in this project has two classes, Alzheimer’s disease (AD) and Cognitive Normal (CN). These images are greyscale and have dimension 256x240 pixels. The dataset contains 30,520 images in total, with the test set having 9000 images, 4460 labelled as AD and 4540 labelled as NC, while the train set has 21,520 images, 10,400 labelled as AD and 11,120 labelled as NC.

Sample MRI images of the ADNI dataset can be seen below:
![AD vs NC Sample](images/AD_NC_sample.png)
*Figure 5: Sample AD and NC MRI*

### Dataset Preparation & Segmentation

The testing set was split into a validation and test set in a 20:80 split.

For the training set the following transformations were made:

- **Resized** to 224 x 224 pixels as done in the original ConvNeXt paper
- **Random Horizontal flip**: randomly horizontally flips the image
- **Random Rotation**: randomly rotates the image by 10 degrees
- **Colour Jitter**: randomly change the brightness, contrast and saturation of an image.
- **Random Affine**: random affine transformations, keeping the center invariant
- **Random Resized Crop**: randomly crop to the given size and aspect ratio
- **Normalised** with a mean and standard deviation of 0.5

Though the paper’s ConvNeXt used significantly more data augmentation, I found that using the following transforms impacted the model’s ability to detect Alzheimer’s and reduced the accuracy of the model. The following transforms are those I tried, but decided not to use in favour of the model’s performance:

- **random erasing**: randomly selects a rectangle region in an image and erases its pixels, however, can remove important features
- **Rand Augment**: randomly applies a specified number of augmentation transformations
- **Random Adjust Sharpness**: randomly adjusts the sharpness of the image
- **CutMix**: randomly selects a pair of input images, cuts and pastes a random patch of pixels from the first image onto second image, however, can create blurred images, especially for those with complex textures[^4]
- **Mixup**: mixes a random pair of input images and their labels, however, can remove important features[^4]

For the testing and validation set, the images were resized to 224 x 224 pixels and normalised with a mean and standard deviation of 0.5.

## Model Training and Experimental Adjustments

## Model Fine-Tuning and Experimental Adjustments

Over 100 experiments were conducted to fine-tune the ConvNeXt architecture and training procedures to achieve optimal binary classification accuracy on the ADNI dataset.

### The ImageNet to ADNI Gap

Since the performance of ConvNeXt doesn’t only rely on its architecture but the training procedures, I tried to keep as true to the original paper’s training procedure for ImageNet as possible. However, I found that many of these training procedures were not suited for training ConvNeXt on the relatively smaller ADNI dataset, leading to a decrease in accuracy.

ImageNet has over 1000 object classes with 1.2M training images, whereas ADNI has only 2 classes with 21,520 training images. This large discrepancy in dataset size might limit ConvNeXt’s ability to fully learn rich feature representations. As a result, the model struggles to generalize effectively on the ADNI dataset, leading to a noticeable reduction in accuracy compared to its performance on ImageNet.

Another point is that the ADNI dataset is comprised of greyscale images, with 1 channel inputs, whereas ImageNet is RGB images with 3 channels. As the ConvNeXt structure allows changing of the input channels, I changed it from 3 to 1 for the training of the ADNI dataset. This change while necessary for data compatibility, may a potential source of performance decrease, as the paper’s original configuration of ConvNeXt was not developed or tested on a single-channel input.

Initial experiments used ConvNeXt-Tiny (ConvNeXt-T), but an increase in performance was observed upon switching to the slightly larger ConvNeXt-Small (ConvNeXt-S), which was used for the final model. ConvNeXt-S shares the same block structure as ConvNeXt-T but incorporates deeper feature extraction in stage 3 with [3, 3, 27, 3] blocks, where ConvNeXt-T had [3, 3, 27, 9] blocks.

During experiments to fine tune the accuracy of the model, I tried using a validation set by splitting the training set in an 80:20 split, however I found that this only decreased accuracy. This could be due to the fact that ConvNeXt was created to run well on larger datasets, and with a smaller dataset like ADNI, performance seems to decrease. I also tried splitting the test set in an 80:20 split to create a validation set, which I implemented in my final model.

In an effort to manage the class imbalance and dataset size difference, I also conducted experiments with different loss functions:

1. Cross-Entropy Loss: The baseline for classification tasks. Had slightly worse performance compared to label smoothing
2. Label Smoothing: Implemented as per the ConvNeXt paper to regularize the model and prevent overconfident predictions.
3. Weighted Cross-Entropy Loss: Used to address any potential class imbalance between the AD and NC groups, helping the model focus on the minority class. I however did not see any improvement in accuracy.

As a result of this experimentation, I chose Label Smoothing as my loss function.

### Hyperparameter Tuning

Given the mismatch in dataset scale, extensive experimentation was required to stabilize training and maximize performance.

| Hyperparameter | Tested Value Range | Final Value |
|----------------|--------------------|-------------|
| Learning Rate  | 0.01 to 0.1        | 0.001       |
| Batch size     | 64 to 256          | 128         |
| Weight Decay   | 0.01 to 0.1        | 0.05        |
| Smoothing      | 0.0 to 0.9         | 0.01        |
| Drop path rate | 0.0 to 0.1         | 0.1         |
| epochs         | 10 to 100          | 60          |

In terms of the learning rate scheduler, I used Cosine Annealing with linear warmup over 5 epochs. This was done to try match original paper, but scaled down to match the number of epochs I had. I also tried various milestones for the linear warmup, and found 5 to be good milestone.

## Results and Analysis

Figure 6 shows the training and validation loss across the 60 epochs. The training loss steadily decreases across epochs, however the validation loss seems to slightly increase.
![loss plot](images\train_val_loss.png)
*Figure 6: Training and validation loss over 60 epochs*

After training for 60 epochs, the model returned an accuracy of 76.78% on the test set. The model's validation accuracies can be seen in figure 7.

![accuracies plot](images\val_accuracy.png)
*Figure 7: validation accuracy over 60 epochs*

**Classification Report:**

```Notepad
              precision    recall  f1-score   support

          NC       0.69      0.92      0.79      3400
          AD       0.90      0.63      0.74      3800

    accuracy                           0.77      7200
   macro avg       0.80      0.78      0.77      7200
weighted avg       0.80      0.77      0.76      7200
```


![confusion matrix](images\Confusion_Matrix.png)
*Figure 8: confusion matrix*

### Analysis of accuracy on a patient level

Though the model had an accuracy of 76.78% accuracy on the test set, that accuracy was on a per image slice. This project aims to classify Alzheimer's, which is done at a per patient level. Thus I carried out analysis of the model's accuracy when diagnosing patients with Alzheimers. First I split the test set into a list of per-patient subset datasets. Each patient in the test set had 20 MRI slices each. Then I evaluated the models performance per patient by averaging slice predictions, and computed ROC + AUC for AD vs NC classification. The result of this evaluation can be seen below.

**Per Patient Classification Report:**

```Notepad
              precision    recall  f1-score   support

          NC       0.72      0.96      0.82       227
          AD       0.94      0.62      0.75       223

    accuracy                           0.79       450
   macro avg       0.83      0.79      0.79       450
weighted avg       0.83      0.79      0.79       450
```

![roc curve](images\predict_roc_curve.png)
*Figure 9: per-patient roc curve*

Per patient, the model had an AUC of 0.872.

![confusion matrix patient](images\predict_Confusion_Matrix.png)
*Figure 10: confusion matrix for per patient analysis*

![patient slice predictions NC](images\patient_slices_NC.png)
*Figure 11: NC patient predictions over all 20 slices*

![patient slice predictions AD](images\patient_slices_AD.png)
*Figure 12: AD patient predictions over all 20 slices*

From this analysis, we can see that when the model is classifying Alzheimer's on a patient level, its accuracy increases to 79.3%.

Thus this project successfuly performs binary classification of 2D brain MRIs the ADNI dataset between patients with Alzheimer's disease (AD) and subjects with Normal Cognition (NC) with an accuracy of 0.793 on the test set using a ConvNeXt architecture.

## Dependencies and Usage

This project was run on the UQ rangpur cluster on a NVidia A100 vGPUs with at least 10GB of GPU memory.

This project was run with the following libraries and dependencies:

- **Python** : 3.9.23
- **Pytorch** : 2.8.0
- **torchvision** : 0.23.0
- **numpy** : 2.0.2
- **timm** : 1.0.20

The dataset should have the following structure on your own machine:

```Notepad
ADNI/
|──AD_NC/
├──── train/
│     ├──── AD/
│     │     ├──── <patientID>_<sliceNumber>.jpeg
│     │     └──── ...
│     └──── NC/
│           ├──── <patientID>_<sliceNumber>.jpeg
│           └──── ...
├──── test/
│     ├──── AD/
│     │     ├──── <patientID>_<sliceNumber>.jpeg
│     │     └──── ...
│     └──── NC/
│           ├──── <patientID>_<sliceNumber>.jpeg
│           └──── ...

```

Using miniconda, run the following command to install all dependencies

```bash
$ conda install pytorch torchvision numpy timm
```

To run the training and testing script:

```bash
$ python train.py -n 60 -l 0.001 -b 128 -w 0.05 -s 5 -m 0.01 -d 0.1
```

To run the example usage of the trained model:

```bash
$ python predict.py 
```

*Note:* you will need to update the variables `train_set_location`, `test_set_location` in `dataset.py` and `checkpoint_location` in `predict.py` to their respective locations on your machine before running the scripts.

## Future Steps

As an extension to this project the following actions could be experimented on and analysed to determine whether they could improve the accuracy:

- Try on a larger scale MRI dataset
- Try combining datasets (OASIS and ADNI) for training for a bigger test set and more variation in images
- Try stacking images to mimic 3 channel input or duplicating same image for 3 channel input
- Evaluate whether early stopping has any impact on accuracy
- Try a larger ConvNeXt models (large, extra large)
- Try a more advanced model, such as Med Mamba

## References

[^1]: Alzheimer's Association, “What Is Alzheimer’s Disease?,” Alzheimer’s Association, 2025. <https://www.alz.org/alzheimers-dementia/what-is-alzheimers>

[^2]: Z. Liu et al., “A ConvNet for the 2020s,” Mar. 2022. Available: <https://arxiv.org/pdf/2201.03545>

[^3]: GeeksforGeeks, ConvNeXt. 2025. Available: <https://www.geeksforgeeks.org/computer-vision/convnext/>

[^4]: L. Monigatti, “Cutout, Mixup, and Cutmix: Implementing Modern Image Augmentations in PyTorch | Towards Data Science,” Towards Data Science, Apr. 14, 2023. <https://towardsdatascience.com/cutout-mixup-and-cutmix-implementing-modern-image-augmentations-in-pytorch-a9d7db3074ad/>