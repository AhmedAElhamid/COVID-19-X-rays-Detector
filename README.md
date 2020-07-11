# COVID-19 X-rays Detector

## Introduction

COVID-19 (coronavirus disease 2019) is a highly infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The first cases were reported in Wuhan; China, in December 2019 before the rapid global spread. The outbreak was subsequently declared as a pandemic on 11th March 2020.
The chest radiographs of patients infected by the novel coronavirus demonstrate characteristic pneumonia-like patterns that can help in the diagnosis, according to a case report by the Chinese Center for Disease Control and prevention published in the New England Journal of Medicine.

Therefore, the main purpose of this project was to help diagnosing COVID-19 from Chest X-ray in Resource Limited Environments.

The project aims to classify chest X-ray images to 3 classes (COVID-19,Viral pneumonia,Normal), using deep convolutional neural networks. The model is trained on the COVID-19 RADIOGRAPHY DATABASE which was published by A team of researchers from Qatar University, Doha, Qatar and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors and was Winner of the COVID-19 Dataset Award. This dataset consists of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images, there are 219 COVID-19 positive images, 1341 normal images and 1345 viral pneumonia images.



## Installations:

Install dependencies using requirements.txt

pip install -r requirements.txt



## Dataset

The COVID-19 RADIOGRAPHY DATABASE
https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

Lets see a sample of the data


![image](https://user-images.githubusercontent.com/42896530/87225009-08471200-c38a-11ea-90bb-e63201a5c434.png)


## Usage:

the program can take any X-ray image to a patient chest and classify it to one of the 3 clases
<br/>1)COVID-19
<br/>2)Viral pneumonia
<br/>3)Normal

I included a tutorial that explains the steps i used to prepare the data and build the model and also included python file that is ready to use with the pretrained model you just have to change the directory to where your data is, i have choosen this model specificaly since it scores the best accuracy Which is about 97% test accuracy and exceeded 99% in training accuracy and i will still try to increase it further when the dataset is updated with more cases.
## Model performance

the model accuracy and loss for the training and the validating set. These plots show the accuracy and loss values for the epochs 1-100. Note that no random seed is specified for this notebook.

![image](https://user-images.githubusercontent.com/42896530/87225002-01b89a80-c38a-11ea-871c-ce9097aef354.png)

## Output
this is the prediction of the sample we taken from the dataset after we trained the model

![image](https://user-images.githubusercontent.com/42896530/87225013-0da45c80-c38a-11ea-9a66-67df5609e284.png)

## Expected outcomes
My expected outcome from this project is that it would give physicians an edge and allow them to act with more confidence while they wait for the analysis of a radiologist by having a digital second opinion confirm their assessment of a patient's condition. And also help people in resource limited environments as the x-ray imaging is the most widely available imaging technique


```python

```
