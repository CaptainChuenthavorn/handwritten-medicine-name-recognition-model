import os
# specify the folder path
folder_path = "01_init\\train_test_data\\test\\amikacin"

# get a list of all the files in the folder with a .jpg extension
file_list = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# print the path to each file in the list
for file in file_list:
    file_path = os.path.join(folder_path, file)
    print(file_path)