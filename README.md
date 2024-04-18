# Prediction & Segmentation of Bacteria Images with U-Net & AlexNet Architectures

## Overview
**Using U-Net and AlexNet-based algorithms**, we analyze and classify large datasets of microscopic images of bacteria. We use a pre-trained VGG16 model to optimize the prediction and segmentation quality of our algorithms. We receive datasets of bacterial images, and through machine learning algorithms, we enable an assessment of the dataset's quality. Additionally, we perform segmentation and prediction operations on the images, which allows us to differentiate the images into bacterial families such as gram-positive and gram-negative bacteria. Furthermore, we predict the type of bacteria using the AlexNet architecture 

** - Prediction of belonging to gram-positive and gram-negative** - stands at 100% accurate prediction.
** - Prediction of the type of bacteria** - ranges between 70-85 percent success rate.

The various histograms we present demonstrate the qualities of the dataset we used in the project. Additionally, these quality statistics indicate the success of the project in the segmentation and identification processes. They represent the quality of our dataset evaluation, thus enabling us to enhance the capabilities of our models accordingly.

**The histogram curve represents the number of bacteria in an image for an entire dataset. This allows us to estimate the numerical quantity of bacteria in the image so that we can understand for which quantity of bacteria we can achieve the highest quality results in the identification process.**

![image](https://github.com/amitgal21/Final_Project-Prediction-Segmentation/assets/101315285/48863288-8dba-4464-9aa4-0050fe8ef7de) 

**In the histogram before us, it is clear that the common contrast values are up to 20, indicating images with similar brightness among the different bacteria. There is a noticeable decrease in frequency as the contrast values increase, suggesting fewer images with high contrast, indicative of Gram-negative bacteria. This helps us more efficiently evaluate the Gram category of the bacteria. Indeed, it can be concluded that Gram-positive bacteria will have lower contrast than Gram-negative bacteria**

![image](https://github.com/amitgal21/Final_Project-Prediction-Segmentation/assets/101315285/c8fb2239-0d72-4c39-92a2-36480d5c03c2)

**The histogram provided illustrates the distribution of texture correlation in images. Correlation is a statistical measure describing the relationship and mutual dependence between two variables. In the context of image texture, a high correlation value indicates a strong relationship between adjacent pixels, suggesting a uniform and consistent texture.
 The correlation values in the samples range from 0.92 to 1.00. There is very little frequency at lower correlation values, and a significant increase in frequency as the correlation approaches 1.00. The peak frequency is near the value of 1.00, indicating a very uniform texture in most of the images examined.**

 ![image](https://github.com/amitgal21/Final_Project-Prediction-Segmentation/assets/101315285/127e55da-0a24-4aec-8fc8-29a019c15d77)


## Motivation 
Quality classification and segmentation operations on bacteria are essential in the field of **microbiology**, the **pharmaceutical industry** , and **disease identification**. The use of advanced machine learning techniques ensures improvement and progress in these areas.


## Methodology

Our project focuses on developing advanced deep learning architectures for segmentation and prediction tasks in medical imaging. We are utilizing UNet for segmentation and AlexNet for prediction, with the VGG16 model enhancing our algorithms’ learning capabilities.

**Step 1: Data Preparation**
Data Collection: We will collect medical images from open and verified databases.
Image Processing: Initial processing will standardize the image size and enhance contrast.
**Step 2: Model Development**
UNet Development: We will construct a UNet model for segmenting relevant features in the images.
AlexNet Development: We will use AlexNet to predict specific medical conditions or features.
Model Training: The models will be trained on training data, using VGG16 as a pretrained model to refine features.
**Step 3: Testing and Evaluation**
Validation: We will evaluate the models' performance on validation and test data.
Tuning: Parameters will be tuned to improve performance based on outcomes.
**Step 4: Publishing and Sharing**
Documentation: Detailed documentation will be written for future use and community benefit.
Sharing: The code and findings will be shared on GitHub to foster collaborative contributions.
This project is designed to serve as a foundation for future developments and to make a tangible impact on the medical and research community.

## Technologies and Dependencies
* Python 3.9.0
* absl-py                      2.1.0
* astunparse                   1.6.3
* certifi                      2024.2.2
* charset-normalizer           3.3.2
* contourpy                    1.2.0
* cycler                       0.12.1
* flatbuffers                  24.3.7
* fonttools                    4.50.0
* gast                         0.5.4
* google-pasta                 0.2.0
* grpcio                       1.62.1
* h5py                         3.10.0
* idna                         3.6
* importlib_metadata           7.0.2
* importlib_resources          6.4.0
* keras                        3.1.1
* kiwisolver                   1.4.5
* libclang                     18.1.1
* Markdown                     3.6
* markdown-it-py               3.0.0
* MarkupSafe                   2.1.5
* matplotlib                   3.8.3
* mdurl                        0.1.2
* ml-dtypes                    0.3.2
* namex                        0.0.7
* numpy                        1.26.4
* opt-einsum                   3.3.0
* optree                       0.10.0
* zipp                         3.18.1
* google colab

## Installation

Go to Book Project Part B - 8 - Installation Guide

## The Project's Book

The attached book details the basic theories and methodologies used in the prediction project and segmentation actions on pathogens, with an emphasis on using CNNs that perform well with images, allowing us to assess quality features critical to industry experts. The book serves as a comprehensive guide, offering insight into the challenges we faced, providing comprehensive theoretical material on medical methods for identifying bacteria upon which we rely. Additionally, it enriches readers' understanding with advanced machine learning applications in the completeness of academic research.

## Contributors

Amit Shitrit
Cyrine Salame

Special thanks to **Dr.Zeev Frenkel** for guidance and support.

## License

This project is licensed under the terms of the [MIT License](LICENSE).



  


















  




