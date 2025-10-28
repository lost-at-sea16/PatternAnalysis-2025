# Classification of Alzheimer's Disease using ConvNeXt

---

Student Number: 46990716
Student Name: Sophia Gleeson

## Table of contents

[Introduction](#introduction)
[ConvNeXt Overview](#convnext-overview)
[Dependencies](#dependencies)
[Usage](#usage)
[Dataset](#dataset)

## Introduction

Alzheimer's disease is an irreversible neurodegenerative disease caused by damage and deterioration of brain tissue, affects a person’s cognitive functions[^R1]. Given that Alzheimer's disease affects millions of people worldwide, the ability to accurately and effectively diagnose the disease is critical for earlier diagnosis, treatment planning and efficiency and patient care.

Though brain imaging is only one of the critical diagnostic steps, it is important to be able to identify even subtle changes in the brain.

This project aims to perform binary classification of 2D brain MRIs from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset between subjects with Alzheimer's disease (AD) and subjects with Normal Cognition (NC). Using one of the latest vision models, ConvNeXt, I aim to achieve a minimum accuracy of 0.8 on the test set.

## ConvNeXt Overview


## Dependencies

This project was run with the following libraries and dependencies:

- **Python** : 3.9.23
- **Pytorch** : 2.8.0
- **torchvision** : 0.23.0
- **numpy** : 2.0.2S
- **timm** : 1.0.20


## Usage

## Dataset

The ADNI dataset used in this project has two classes, Alzheimer’s disease (AD) and Cognitive Normal (CN). These images are greyscale and have dimension 256x240 pixels. The dataset contains 30,520 images in total, with the test set having 9000 images, 4460 labelled as AD and 4540 labelled as NC, while the train set has 21,520 images, 10,400 labelled as AD and 11,120 labelled as NC. 

Sample MRI images of the ADNI dataset can be seen below:
![AD Sample](images/sample_AD.jpeg)
*Sample AD MRI*

![NC Sample](images/sample_NC.jpeg)  
*Sample NC MRI*

### Dataset Preperation

For the training set the following transformations were made:

- Resized to 224 x 224 pixels
- Normalised with a mean and standard deviation of 0.5
- Horizontal flip
- Random Rotation
- Colour Jitter
- Random Affine
- Random Augmentation
- Random Erasing

For the testing set, the images were resized to 224 x 224 pixels and normalised with a mean and standard deviation of 0.5.

## Training

The model was trained from scratch for 60 epochs on the training set, 

Loss Function: 

Learning rate:

Scheduler: 

Optimiser




## Testing

## Results?

## Observations?

## Conclusion

## References

[^R1]:<https://www.alz.org/alzheimers-dementia/what-is-alzheimers>
