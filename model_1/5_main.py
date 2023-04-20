import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Read the data from CSV file
data = pd.read_csv('data_train_test_info_captain.csv')

# Extract the features and labels
X = np.array(data.drop(['label'], axis=1))
y = np.array(data['label'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print out shapes of training and testing sets
print(f'Training data shape: {X_train.shape}, labels shape: {y_train.shape}')
print(f'Testing data shape: {X_test.shape}, labels shape: {y_test.shape}')

# Reshape the training and testing data to match the expected input shape of the model
X_train = X_train.reshape((X_train.shape[0], 150, 300, 1))
X_test = X_test.reshape((X_test.shape[0], 150, 300, 1))

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 300, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model with loss function, optimizer, and evaluation metric
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training set for a specified number of epochs
model.fit(X_train, y_train, epochs=10)

# Evaluate the model on the test set
model.evaluate(X_test, y_test)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')