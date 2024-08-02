# Install software 
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import layers,models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
diabetes_dataset = load_diabetes()

# Set dataframe
diabetesDataFrame = pd.DataFrame(diabetes_dataset.data, columns=diabetes_dataset.feature_names)

# Add respone as a column 
diabetesDataFrame['Response'] = diabetes_dataset.target

# Print adding of class
print(diabetesDataFrame['Response'].value_counts())

# Extract features and 'class'
X = diabetesDataFrame.drop(columns='Response', axis= 1)
Y = diabetes_dataset.target

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(X.shape, X_train.shape, X_test.shape)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, Y_train)

# Predict on training data
Y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(Y_train, Y_train_pred)
train_r2 = r2_score(Y_train, Y_train_pred)
print(f"Training MSE: {train_mse}")
print(f"Training R²: {train_r2}")

# Predict on testing data
Y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(Y_test, Y_test_pred)
test_r2 = r2_score(Y_test, Y_test_pred)
print(f"Testing MSE: {test_mse}")
print(f"Testing R²: {test_r2}")

# Test model 
input_data = ()

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

threshold = 100

if (prediction[0] < threshold):
  print('The Persons diabetes is well controlled')
else:
  print('The Persons diabetes is not well controlled')

