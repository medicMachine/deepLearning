import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load the Diabetes dataset
diabetes = load_diabetes()
# Extract the features and target variable
X = diabetes.data
y = diabetes.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and fit the LASSO model
lasso = Lasso(alpha=1.0)  # Alpha is the regularization parameter
lasso.fit(X_train, y_train)

# Make predictions
y_pred = lasso.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error for diabetes dataset: {mse}")
print(f"R^2 Score for diabetes dataset: {r2}")

# Plot the coefficients
pyplot.figure(figsize=(10, 6))
pyplot.bar(range(len(lasso.coef_)), lasso.coef_)
pyplot.title('LASSO Coefficients')
pyplot.xlabel('Feature Index')
pyplot.ylabel('Coefficient Value')
pyplot.show()




