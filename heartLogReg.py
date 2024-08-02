# Install programmes
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import layers,models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset from an Excel file
file_path = '/Users/robertadunn/Library/Mobile Documents/com~apple~Numbers/Documents/heart_disease_data.xlsx'
heart_dataset = pd.read_excel(file_path)

# Inspect dataset
print(heart_dataset.head())

# Check for missing values
print(heart_dataset.isnull().sum())

#Class value interpretations #1 --> Defective Heart #0 --> Healthy Heart

# Identify imbalance in dataset
print(heart_dataset['class'].value_counts())

#Visualise using countplot
sns.countplot(x='class', data= heart_dataset)

# Create for loops to look at distribution of columns
for column in heart_dataset:
    print(column)

for column in heart_dataset:
    sns.displot(x=column, data=heart_dataset)

# A lot of variation in positive and negative skewness

# Pair plot 
sns.pairplot(data=heart_dataset)

# Extract features and 'class'
X = heart_dataset.drop(columns='class', axis= 1)
Y= heart_dataset['class']
print(X.shape)
print(Y.shape)

# Check the columns in X
print(X.columns)

# Standardize the features
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state=0)
print(X.shape, X_train.shape, X_test.shape)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Use model for prediction in training dataset 
X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_accuracy)

# Use model for prediction in testing dataset
X_test_prediction = model.predict(X_test)
testing_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy of Testing dataset; ', testing_accuracy)

# Test model 

input_data = ()

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

