import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import csv

# Read the data from CSV file
data = pd.read_csv('01_init\\train_test_data\\data_train_test_info.csv')

# Extract the file paths and labels
X = np.array([os.path.join('data', f) for f in data['File Name']])
y = np.array([1 if t == 'amikacin' else 0 for t in X])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print out shapes of training and testing sets
print(f'Training data shape: {X_train.shape}, labels shape: {y_train.shape}')
print(f'Testing data shape: {X_test.shape}, labels shape: {y_test.shape}')

# Define a function to load and preprocess an image file
def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [150, 300])
    return img

def resize_image(img, target_size=(150, 300)):
    resized_img = tf.image.resize(img, target_size)
    return resized_img

def normalize_image(img):
    normalized_img = tf.cast(img, tf.float32) / 255.0
    return normalized_img

# Define a function to preprocess a batch of image files
@tf.function
def preprocess_image(file_path):
    img = load_image(file_path)
    img = resize_image(img)
    img = normalize_image(img)
    return img

# Create a dataset for training and testing sets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(32)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


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
model.fit(train_dataset, epochs=10)

# Evaluate the model on the test set
model.evaluate(test_dataset)

# Evaluate the model on the test set
# Evaluate the model on the test set
model.evaluate(X_test, y_test)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')


#X_TEST PART
# Create a list to store the predicted results
predicted_results = []

# Iterate over the images in X_test
for i in range(len(X_test)):
    image = X_test[i]
    image_array = image.reshape((1, 150, 300, 1))
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    if predicted_class == 0:
        predicted_result = "Amikacin"
    else:
        predicted_result = "Cefazolin"
    predicted_results.append(predicted_result)
    
# Write the predicted results to a CSV file
with open('predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Index', 'Predicted Result'])
    for i in range(len(predicted_results)):
        writer.writerow([i, predicted_results[i]])