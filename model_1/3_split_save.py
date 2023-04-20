import os
import random
import csv
from shutil import copyfile
from sklearn.model_selection import train_test_split

class_names = ['amikacin', 'cefazolin']
num_classes = len(class_names)

# Set the path to the directory containing the original dataset
data_dir = '01_init\\binarized_image'

# Set the path to the directory where you want to store the split dataset
split_dir = '01_init\\train_test_data'

# Define the path to the CSV file where the file names and types will be saved
csv_file = '01_init\\train_test_data\\data_train_test_info.csv'
# Set the percentage of images to use for testing
test_size = 0.2

# Create a directory for each class in the split dataset
for class_name in class_names:
    os.makedirs(os.path.join(split_dir, 'train', class_name), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'test', class_name), exist_ok=True)

# Iterate over each class in the original dataset
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    filenames = os.listdir(class_dir)
    random.shuffle(filenames)
    split_index = int(len(filenames) * test_size)
    test_filenames = filenames[:split_index]
    train_filenames = filenames[split_index:]
    
    # Copy the training images to the split dataset directory
    for filename in train_filenames:
        src = os.path.join(class_dir, filename)
        dst = os.path.join(split_dir, 'train', class_name, filename)
        copyfile(src, dst)
    
    # Copy the testing images to the split dataset directory
    for filename in test_filenames:
        src = os.path.join(class_dir, filename)
        dst = os.path.join(split_dir, 'test', class_name, filename)
        copyfile(src, dst)
        
# Load the training and testing data into memory
X_train, y_train = [], []
X_test, y_test = [], []

for i, class_name in enumerate(class_names):
    train_dir = os.path.join(split_dir, 'train', class_name)
    test_dir = os.path.join(split_dir, 'test', class_name)
    
    # Load the training data
    for filename in os.listdir(train_dir):
        img_path = os.path.join(train_dir, filename)
        X_train.append(img_path)
        y_train.append(i)
    
    # Load the testing data
    for filename in os.listdir(test_dir):
        img_path = os.path.join(test_dir, filename)
        X_test.append(img_path)
        y_test.append(i)

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Print the shapes of the training and testing data
print(f'Training data shape: {len(X_train)}, labels shape: {len(y_train)}')
print(f'Validation data shape: {len(X_val)}, labels shape: {len(y_val)}')
print(f'Testing data shape: {len(X_test)}, labels shape: {len(y_test)}')
