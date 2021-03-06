# Image Recognition
 This project explores food image classification with convolutional neural networks (CNNs) for better image labelling and classification by dish, which is the foundation to determine the nutritional value of the dish. The objective of this assignment is to, given an image of a dish as the input to the model, output the correct categorization of the food image. The 10 dishes that the model will recognize are **beet salad, ceviche, chicken quesadilla, creme brulee, garlic bread, macaroni and cheese, miso soup, pad Thai, shrimp and grits, and sushi.**

# Dataset
The dataset is downlaoded from Kaggle (https://www.kaggle.com/dansbecker/food-101). 

# Approach
There will be three final models in this project, one built from scratch, and two built from pre-trained weights learned on a larger image dataset (transfer learning). For the build from scratch model, I will start with 1 convolutional and 1 fully-connected layer. This model will be 
optimized through the use of **network topology, data augmentation, dropout, learning rate, 
batches and epochs, optimization and loss, and regularization.** For the pre-trained model, I 
will be using **Vgg16 and InceptionV3.** This pre-trained model will be optimized through **data 
augmentation, unfreezing one layer or more for fine-tuning.** These three final models will be 
evaluated using test images. A detailed analysis of the model performance will be done to 
determine the best model. The best model will be used to further proof the accuracy of 
image recognition by predicting images that was never seen by the model from google 
images.

# Best Model
Model Name | Model Number | Testing Accuracy
------------ | ------------- | -------------
Build from scratch | 16 | 72.50%
**Vgg16** | **26** | **83%**
Inception V3 | 35 | 76%

After selecting the best model from each training phase, the best model is the Vgg16 model 
which yields the highest testing accuracy at **83.00%.** This is 10.50% higher than the build 
from scratch model and 7% higher than the InceptionV3 model. 

# Prediction with the best model
After we have selected the best model (vgg16, model 26), I downloaded a few images from 
the google images to test the prediction of the images with the model. The images that I 
have downloaded are not seen by the model before to ensure a fair test. The images will be 
converted to size 150x150 because this is the similar size that we used to train the model. 
All the images will be converted to a numpy array so that the model is able to “read” it. 

## Prediction 1 
![Image of Prediction 1](https://github.com/victorjongsoon/Image-Recognition/blob/main/Github%20Images/Prediction%201.PNG)
Through our naked eyes, we can identify this image to be a **sushi** briefly. Similarly, with the 
model, the model can predict this image to be a sushi and the prediction is **99.99%** accurate. 
This shows that the model is very accurate in identifying the images that we have trained.

## Prediction 2
![Image of Prediction 2](https://github.com/victorjongsoon/Image-Recognition/blob/main/Github%20Images/Prediction%202.PNG)
Through our naked eyes, we might have trouble recognizing this image to be a **sushi** 
because a typical sushi will not have a cartoon image on the sushi rice. However, if we take 
a longer time to analyze the image, we can guess that this is a sushi image because of the 
seaweed surrounding the rice. This is a difficult task for the model because not only of the 
cartoon image, there are different foods surrounding the sushi, which makes it harder for the 
model to predict. In this case, the model is still able to predict that this image is a sushi with 
**75.00%** accuracy. This is a very accurate prediction.

## Prediction 3
![Image of Prediction 3](https://github.com/victorjongsoon/Image-Recognition/blob/main/Github%20Images/Prediction%203.PNG)
Through our naked eyes, we can identify this image to be **garlic bread** at first glance. 
Similarly, with the model, the model can predict this image to be a garlic bread and the 
prediction is **99.98%** accurate. This shows that the model is very accurate in identifying the 
images that we have trained.

## Prediction 4
![Image of Prediction 4](https://github.com/victorjongsoon/Image-Recognition/blob/main/Github%20Images/Prediction%204.PNG)
Through our naked eyes, we can identify this image to be **miso soup** at first glance. Similarly,
with the model, the model can predict this image to be a miso soup and the prediction is 
**100%** accurate. This shows that the model is very accurate in identifying the images that we 
have trained.

## Prediction 5
![Image of Prediction 5](https://github.com/victorjongsoon/Image-Recognition/blob/main/Github%20Images/Prediction%205.PNG)
Through our naked eyes, we can easily identify this picture to be a bowl of **chicken rice** at 
first glance. However, because we did not train the model to recognize chicken rice, the 
prediction of this picture becomes inaccurate. Interestingly, because of the chicken on the 
chicken rice, the model can predict the closest food that we have trained, which is chicken 
quesadilla at **94.24%** accuracy. It is expected for this prediction to be inaccurate because we 
did not train the model to recognize chicken rice. 

# Skillset required for the project (Deep Learning)
* Python (Numpy, Pandas etc.)
* Data Modeling and Evaluation
* Transfer Learning (Resnet50, Vgg16, InceptionV3 etc.)
