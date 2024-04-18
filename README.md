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











  




