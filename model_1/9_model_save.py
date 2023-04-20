import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import csv


import tensorflow as tf
# Set up paths to image data
data_dir = '01_init\\binarized_image'
class_names = ['amikacin', 'cefazolin']
num_classes = len(class_names)

# Load images and labels into numpy arrays
X = []
y = []
for i, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for filename in os.listdir(class_dir):
        img = Image.open(os.path.join(class_dir, filename))
        # print(img)
        img = np.array(img)
        X.append(img)
        y.append(i)

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print out shapes of training and testing sets
print(f'Training data shape: {X_train.shape}, labels shape: {y_train.shape}')
print(f'Testing data shape: {X_test.shape}, labels shape: {y_test.shape}')

# X_train.tofile('01_init\\train_test_data\\X_train.csv', sep = ',')
# y_train.tofile('01_init\\train_test_data\\y_train.csv', sep = ',')
# X_test.tofile('01_init\\train_test_data\\X_test.csv', sep = ',')
# y_test.tofile('01_init\\train_test_data\\y_test.csv', sep = ',')

# Reshape the training and testing data to match the expected input shape of the model
X_train = X_train.reshape((80, 150, 300, 1))
X_test = X_test.reshape((20, 150, 300, 1))

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

# save the trained model
model.save('my_model.h5')

# Evaluate the model on the test set
model.evaluate(X_test, y_test)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')