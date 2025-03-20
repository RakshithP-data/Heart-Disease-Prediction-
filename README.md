# Heart-Disease-Prediction-
🫀 We will utilize machine learning models to predict the likelihood of a patient having heart disease based on various features such as cholesterol levels and chest pain type. For this purpose, we’ll be working with a sample dataset to explore and understand how different machine learning models can be applied and to gain hands-on experience.

# 🫀 Heart Disease Prediction Using Machine Learning and Deep Learning

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction 

Predicting the condition of a patient in the case of __heart disease__ is important. It would be good if a patient could get to know the condition before itself rather than visiting the doctor. Patients spend a __significant amount__ of time trying to get an appointment with doctors. After they get the appointment, they would have to give a lot of tests. 
It is also important that a doctor be present so that they could treat them. To make things worse, the tests usually take a long time before __diagnosing__ whether a person would suffer from heart disease. However, it would be handy and better if we automate this process which ensures that we save a lot of time and effort on the part of the doctor and patient. 

## Machine Learning and Data Science 

With the aid of machine learning, it is possible to get to know the condition of a patient whether he/she would have heart disease or not based on a few features such as glucose levels and so on. Using machine learning algorithms, we are going to predict the chances of a person suffering from heart disease. In other words, we would be using various machine learning models for the prediction of the chances of heart disease in a patient. 

## Data 

Some of the input features that would be considered in this example are __blood pressure__, __chest pain type__ and __cholestrol levels__. This is just a sample dataset with 303 rows. This is created just to understand different classification machine learning algorithms and sklearn implementation of them. Below is the link where you can find the source of the data. Thanks to __'kaggle'__ for providing this data. 

https://www.kaggle.com/johnsmith88/heart-disease-dataset

## Exploratory Data Analysis (EDA)

When we are performing EDA, interesting insights could be drawn from the data and understood. Below are the key insights that were found as a result of working with the data. 

* The features __'thalach'__ and __'slope'__ are positively correlated. However, they are not highly correlated.
* Features such as __'age'__ and __'thalach'__ which stands for __maximum heart rate achieved__ are negatively correlated as indicated by the correlation heatmap plot. 
* __'age'__ is negatively correlated with __'restecg'__ as shown in the heatmap plot.
* The features __'cp'__ and __'exang'__ are negatively correlated.
* The feature __'resting blood pressure'__ is somewhat positively correlated with __'age'__ as shown in the plots. 
* The feature __'trestbps'__ that stands for resting blood pressure is somewhat correlated with the feature __'fbs'__ that stands for fasting blood pressure.

## Visualizations 

In this section, we will be visualizing some interests plots and trends in the data which is used by machine learning models to make predictions. We will also evaluate the performance of different machine learning and deep learning models on a given dataset by comparing various metrics. To identify the best version of each model, we will examine their hyperparameters.

## Machine Learning Models 

There were many machine learning models used in the process of predicting the __heart diseases__. Below are the models that were used in the process of predicting heart diseases.

* [__K Nearest Neighbors (KNN)__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [__Logistic Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [__Naive Bayes__](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
* [__Random Forest Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## Outcomes

* Based on the results generated, it could be seen that the __Naive Bayes model__ was performing the best in terms of the __F1 score__, __precision__, and __recall__ respectively. 
* __Naive Bayes model__ was also having the __highest accuracy__ in terms of classifying whether a person would have a __heart disease__ or not.

## Future Scope

* __Additional data__ from many sources could be taken so that the models would be able to predict for __different conditions__ for the patients.
* __More features__ that help determine whether a person would suffer from heart disease could be considered.
* The **Naive Bayes model**, which had the best performance, could be deployed in real-time to provide doctors with faster inference results. This could aid in the diagnosis of whether a person is suffering from heart disease or not. 
* The quick and accurate results from the model could potentially help **doctors** make more timely and effective decisions in treating their patients.

