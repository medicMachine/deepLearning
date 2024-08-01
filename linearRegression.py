# Install and import 
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Diabetes dataset
diabetes = load_diabetes()
# Extract the features and target variable
X = diabetes.data
y = diabetes.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build linear regression model
lreg = LinearRegression()
lreg.fit(X_train, y_train)

# Predict and evaluate
lreg_pred = lreg.predict(X_test)
lreg_mse = mean_squared_error(y_test, lreg_pred)
lreg_r2 = r2_score(y_test, lreg_pred)

# Print MSE and r2 score
print("Linear Regression for diabetes dataset MSE", lreg_mse)
print("Linear Regression for diabetes dataset r2", lreg_r2)
