import os
import csv
from sklearn.model_selection import train_test_split

# Define class names and number of classes
class_names = ['amikacin', 'cefazolin']
num_classes = len(class_names)
# Set the path to the directory containing the original dataset
data_dir = '01_init\\binarized_image'


# Define the path to the CSV file where the file names and types will be saved
csv_file = '01_init\\train_test_data\\data_train_test_info.csv'

# Split the data into training and testing sets
split_dir = {}
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    file_names = os.listdir(class_dir)
    X = [os.path.join(class_dir, file_name) for file_name in file_names]
    y = [class_name] * len(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    split_dir.update({os.path.join(data_dir, file_name): 'train' for file_name in X_train})
    split_dir.update({os.path.join(data_dir, file_name): 'test' for file_name in X_test})

# Write the file names and types to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['File Name', 'Type'])
    for file_name, file_type in split_dir.items():
        writer.writerow([file_name, file_type])
