# Install dataset for breast cancer
import pandas as pd
import numpy as np
import keras as keras 
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
breast_cancer_dataset = load_breast_cancer()

# Form a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)

# Add column as class
data_frame['class'] = breast_cancer_dataset.target 

# Print adding of class
print(data_frame.tail())

# Check missing values
print(data_frame.describe())

# Checking balance of dataset
print(data_frame.groupby("class").mean())

# Seperation of dataset 
X = data_frame.drop(columns='class', axis= 1)
Y = data_frame['class']
print(Y)
print(X)

# Split to train and test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 2)

# Use scaler to improve accuracy
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Set random seed 
import tensorflow as tf
tf.random.set_seed(3)
from keras import layers, models 

# Building neural network layers
model = models.Sequential([
    tf.keras.layers.Flatten(input_shape=(30,)),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(2, activation="sigmoid")
])


# Compile neural network
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Validate and fit model
history = model.fit(X_train_std, Y_train, validation_split = 0.1, epochs = 10)

# Plotting model accuracy 
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(['Training Data', 'Validation Data']) # Add legend
plt.show()

# Predictions
Y_pred = model.predict(X_test_std)

Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)

# Make a prediction
input_data = (13.71,	20.83,	90.2,	577.9,	0.1189,	0.1645,	0.09366,	0.05985,	0.2196,	0.07451,	0.5835,	1.377,	3.856,	50.96,	0.008805,	0.03029	,0.02488	,0.01448,	0.01486,	0.005412,	17.06,	28.14,	110.6,	897,	0.1654,	0.3682,	0.2678,	0.1556,	0.3196,	0.1151)

input_data_as_numpy = np.asarray(input_data)
input_data_reshape = input_data_as_numpy.reshape(1,-1)
input_data_std = scaler.transform(input_data_reshape)
prediction = model.predict(input_data_std)
print(prediction)

# label prediction
prediction_label = [np.argmax(prediction)]

if(prediction_label[0] == 0):
    print('Tumour is Malignant')

else:
    print('Tumour is Benign')

# Construct heatmap
# Combine features and target into a single DataFrame for easier plotting
data = pd.concat([X, Y], axis=1)
# Compute the correlation matrix
corr = data.corr()

# Plot the heatmap 
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()






