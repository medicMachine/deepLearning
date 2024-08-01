# Install and import 
import numpy as np
import matplotlib.pyplot as pyplot # type: ignore
from sklearn.datasets import load_diabetes # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load the Diabetes dataset
diabetes = load_diabetes()
# Extract the features and target variable
X = diabetes.data
y = diabetes.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build ridge regression
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

# Prediction from ridge regression 
ridge_pred = ridge.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

# Print results 
print("Ridge regression for diabetes data MSE", ridge_mse)
print("Ridge regression for diabetes dataset r2", ridge_r2)


